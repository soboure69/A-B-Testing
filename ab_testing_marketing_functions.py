"""
Fonctions pour l'analyse A/B Testing Marketing
=============================================

Ce module contient toutes les fonctions nécessaires pour analyser les résultats
d'un test A/B dans le contexte marketing, notamment pour comparer les taux de
conversion entre deux versions d'une landing page.

Auteur: Data Science Expert
Date: 2025-08-08
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, chi2_contingency
import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Configuration des graphiques
plt.style.use('default')
sns.set_palette("husl")

def load_and_explore_data(file_path):
    """
    Charge et explore le dataset A/B Testing
    
    Parameters:
    -----------
    file_path : str
        Chemin vers le fichier CSV
        
    Returns:
    --------
    pd.DataFrame : Dataset chargé
    dict : Statistiques descriptives
    """
    # Chargement des données
    df = pd.read_csv(file_path)
    
    # Conversion de la colonne timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Statistiques descriptives
    stats_dict = {
        'shape': df.shape,
        'columns': list(df.columns),
        'data_types': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'unique_users': df['user_id'].nunique(),
        'date_range': (df['timestamp'].min(), df['timestamp'].max()),
        'groups': df['group'].value_counts().to_dict(),
        'landing_pages': df['landing_page'].value_counts().to_dict(),
        'conversion_summary': df['converted'].value_counts().to_dict()
    }
    
    return df, stats_dict

def validate_experiment_design(df):
    """
    Valide la qualité de l'expérimentation A/B
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset A/B Testing
        
    Returns:
    --------
    dict : Résultats de validation
    """
    validation_results = {}
    
    # 1. Vérification de la cohérence groupe/landing_page
    coherence_check = df.groupby(['group', 'landing_page']).size().unstack(fill_value=0)
    validation_results['coherence_matrix'] = coherence_check
    
    # Détection des incohérences
    control_new_page = coherence_check.loc['control', 'new_page'] if 'new_page' in coherence_check.columns else 0
    treatment_old_page = coherence_check.loc['treatment', 'old_page'] if 'old_page' in coherence_check.columns else 0
    
    validation_results['inconsistencies'] = {
        'control_with_new_page': control_new_page,
        'treatment_with_old_page': treatment_old_page,
        'total_inconsistencies': control_new_page + treatment_old_page
    }
    
    # 2. Équilibrage des groupes
    group_sizes = df['group'].value_counts()
    validation_results['group_balance'] = {
        'control_size': group_sizes.get('control', 0),
        'treatment_size': group_sizes.get('treatment', 0),
        'balance_ratio': group_sizes.get('treatment', 0) / group_sizes.get('control', 1) if group_sizes.get('control', 0) > 0 else 0
    }
    
    # 3. Vérification des doublons
    validation_results['duplicates'] = {
        'duplicate_users': df['user_id'].duplicated().sum(),
        'unique_users': df['user_id'].nunique(),
        'total_records': len(df)
    }
    
    return validation_results

def clean_data(df):
    """
    Nettoie le dataset en supprimant les incohérences
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset brut
        
    Returns:
    --------
    pd.DataFrame : Dataset nettoyé
    dict : Rapport de nettoyage
    """
    initial_shape = df.shape
    
    # Suppression des incohérences groupe/landing_page
    df_clean = df[
        ((df['group'] == 'control') & (df['landing_page'] == 'old_page')) |
        ((df['group'] == 'treatment') & (df['landing_page'] == 'new_page'))
    ].copy()
    
    # Suppression des doublons d'utilisateurs (garder le premier)
    df_clean = df_clean.drop_duplicates(subset=['user_id'], keep='first')
    
    cleaning_report = {
        'initial_records': initial_shape[0],
        'final_records': df_clean.shape[0],
        'removed_records': initial_shape[0] - df_clean.shape[0],
        'removal_percentage': ((initial_shape[0] - df_clean.shape[0]) / initial_shape[0]) * 100
    }
    
    return df_clean, cleaning_report

def calculate_conversion_rates(df):
    """
    Calcule les taux de conversion par groupe
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset nettoyé
        
    Returns:
    --------
    dict : Métriques de conversion
    """
    # Calculs par groupe
    conversion_stats = df.groupby('group').agg({
        'converted': ['count', 'sum', 'mean'],
        'user_id': 'nunique'
    }).round(4)
    
    # Aplatissement des colonnes multi-index
    conversion_stats.columns = ['total_users', 'conversions', 'conversion_rate', 'unique_users']
    
    # Calculs globaux
    total_users = df.shape[0]
    total_conversions = df['converted'].sum()
    overall_rate = df['converted'].mean()
    
    # Métriques par groupe
    control_rate = conversion_stats.loc['control', 'conversion_rate']
    treatment_rate = conversion_stats.loc['treatment', 'conversion_rate']
    
    # Calcul du lift
    lift = ((treatment_rate - control_rate) / control_rate) * 100 if control_rate > 0 else 0
    absolute_lift = treatment_rate - control_rate
    
    results = {
        'overall_metrics': {
            'total_users': total_users,
            'total_conversions': total_conversions,
            'overall_conversion_rate': overall_rate
        },
        'by_group': conversion_stats.to_dict('index'),
        'comparison': {
            'control_rate': control_rate,
            'treatment_rate': treatment_rate,
            'absolute_lift': absolute_lift,
            'relative_lift_percent': lift
        }
    }
    
    return results

def descriptive_group_metrics(df):
    """
    Analyse descriptive des performances par groupe avec intervalles de confiance
    et coupes temporelles (jour, heure).

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset nettoyé (colonnes requises: group, converted, timestamp)

    Returns:
    --------
    dict :
        - summary_by_group: DataFrame avec n, conversions, taux, IC95%
        - by_day: DataFrame taux quotidiens par groupe (index=date)
        - by_hour: DataFrame taux horaires par groupe (0-23)
    """
    # Summary par groupe avec IC 95%
    grp = df.groupby('group').agg(n=('converted', 'size'),
                                  conversions=('converted', 'sum'),
                                  rate=('converted', 'mean'))
    # Erreur standard et IC normal approx
    grp['se'] = np.sqrt(grp['rate'] * (1 - grp['rate']) / grp['n']).replace([np.inf, -np.inf], 0)
    grp['ci_low'] = (grp['rate'] - 1.96 * grp['se']).clip(lower=0)
    grp['ci_high'] = (grp['rate'] + 1.96 * grp['se']).clip(upper=1)

    # Tendance quotidienne
    tmp = df.copy()
    tmp['date'] = tmp['timestamp'].dt.date
    by_day = tmp.groupby(['date', 'group'])['converted'].mean().unstack()

    # Tendance horaire
    tmp['hour'] = tmp['timestamp'].dt.hour
    by_hour = tmp.groupby(['hour', 'group'])['converted'].mean().unstack()

    return {
        'summary_by_group': grp.reset_index(),
        'by_day': by_day,
        'by_hour': by_hour
    }

def run_logistic_regression(df, add_time_features=True):
    """
    Régression logistique (multivariée) pour modéliser la conversion.

    Features:
    - Variable principale: group (treatment vs control)
    - Optionnel: effets fixes pour le jour de la semaine et l'heure de la journée

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset nettoyé (colonnes requises: converted, group, timestamp)
    add_time_features : bool
        Si True, ajoute C(day_of_week) et C(hour)

    Returns:
    --------
    dict :
        - model_summary: str (résumé textuel)
        - params: DataFrame des coefficients
        - odds_ratios: DataFrame des OR et IC95%
        - pvalues: Series p-values
    """
    data = df.copy()
    # Encodage binaire du groupe
    data['group_bin'] = (data['group'] == 'treatment').astype(int)
    # Caractéristiques temporelles
    formula = 'converted ~ group_bin'
    if add_time_features:
        data['day_of_week'] = data['timestamp'].dt.dayofweek  # 0=Lundi
        data['hour'] = data['timestamp'].dt.hour
        formula += ' + C(day_of_week) + C(hour)'

    try:
        model = smf.logit(formula=formula, data=data).fit(disp=False)
    except Exception as e:
        return {
            'model_summary': f"Échec de l'ajustement du modèle: {e}",
            'params': pd.DataFrame(),
            'odds_ratios': pd.DataFrame(),
            'pvalues': pd.Series(dtype=float)
        }

    params = model.params.to_frame(name='coef')
    conf = model.conf_int()
    conf.columns = ['ci_low', 'ci_high']
    or_df = pd.DataFrame({
        'odds_ratio': np.exp(model.params),
        'or_ci_low': np.exp(conf['ci_low']),
        'or_ci_high': np.exp(conf['ci_high'])
    })

    return {
        'model_summary': model.summary2().as_text(),
        'params': params.join(conf),
        'odds_ratios': or_df,
        'pvalues': model.pvalues
    }

def perform_statistical_tests(df, alpha=0.05):
    """
    Effectue les tests statistiques pour l'A/B Testing
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset nettoyé
    alpha : float
        Seuil de significativité (défaut: 0.05)
        
    Returns:
    --------
    dict : Résultats des tests statistiques
    """
    # Préparation des données
    control_group = df[df['group'] == 'control']
    treatment_group = df[df['group'] == 'treatment']
    
    n_control = len(control_group)
    n_treatment = len(treatment_group)
    
    conversions_control = control_group['converted'].sum()
    conversions_treatment = treatment_group['converted'].sum()
    
    rate_control = conversions_control / n_control
    rate_treatment = conversions_treatment / n_treatment
    
    # 1. Test Z pour la différence de proportions
    # Proportion poolée
    p_pool = (conversions_control + conversions_treatment) / (n_control + n_treatment)
    
    # Erreur standard
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n_control + 1/n_treatment))
    
    # Statistique Z
    z_stat = (rate_treatment - rate_control) / se if se > 0 else 0
    
    # P-value (test bilatéral)
    p_value_z = 2 * (1 - norm.cdf(abs(z_stat)))
    
    # 2. Test Chi-2 d'indépendance
    contingency_table = pd.crosstab(df['group'], df['converted'])
    chi2_stat, p_value_chi2, dof, expected = chi2_contingency(contingency_table)
    
    # 3. Intervalles de confiance
    # Pour le groupe contrôle
    se_control = np.sqrt(rate_control * (1 - rate_control) / n_control)
    ci_control = (
        rate_control - 1.96 * se_control,
        rate_control + 1.96 * se_control
    )
    
    # Pour le groupe traitement
    se_treatment = np.sqrt(rate_treatment * (1 - rate_treatment) / n_treatment)
    ci_treatment = (
        rate_treatment - 1.96 * se_treatment,
        rate_treatment + 1.96 * se_treatment
    )
    
    # Pour la différence
    se_diff = np.sqrt(se_control**2 + se_treatment**2)
    diff = rate_treatment - rate_control
    ci_diff = (
        diff - 1.96 * se_diff,
        diff + 1.96 * se_diff
    )
    
    results = {
        'sample_sizes': {
            'control': n_control,
            'treatment': n_treatment
        },
        'conversion_rates': {
            'control': rate_control,
            'treatment': rate_treatment,
            'difference': diff
        },
        'z_test': {
            'z_statistic': z_stat,
            'p_value': p_value_z,
            'is_significant': p_value_z < alpha
        },
        'chi2_test': {
            'chi2_statistic': chi2_stat,
            'p_value': p_value_chi2,
            'degrees_of_freedom': dof,
            'is_significant': p_value_chi2 < alpha
        },
        'confidence_intervals_95': {
            'control': ci_control,
            'treatment': ci_treatment,
            'difference': ci_diff
        },
        'statistical_power': calculate_statistical_power(n_control, n_treatment, rate_control, rate_treatment)
    }
    
    return results

def calculate_statistical_power(n1, n2, p1, p2, alpha=0.05):
    """
    Calcule la puissance statistique du test
    
    Parameters:
    -----------
    n1, n2 : int
        Tailles des échantillons
    p1, p2 : float
        Proportions observées
    alpha : float
        Seuil de significativité
        
    Returns:
    --------
    dict : Métriques de puissance
    """
    # Effet observé
    effect_size = abs(p2 - p1)
    
    # Proportion poolée
    p_pool = (n1 * p1 + n2 * p2) / (n1 + n2)
    
    # Erreur standard sous H0
    se_h0 = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    
    # Erreur standard sous H1
    se_h1 = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    
    # Valeur critique
    z_alpha = norm.ppf(1 - alpha/2)
    
    # Puissance (approximation)
    z_beta = (effect_size - z_alpha * se_h0) / se_h1 if se_h1 > 0 else 0
    power = norm.cdf(z_beta)
    
    return {
        'effect_size': effect_size,
        'statistical_power': max(0, power),
        'minimum_detectable_effect': z_alpha * se_h0
    }

def visualize_results(df, conversion_metrics, statistical_results, save_plots=False):
    """
    Crée les visualisations pour l'analyse A/B Testing
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset nettoyé
    conversion_metrics : dict
        Métriques de conversion
    statistical_results : dict
        Résultats des tests statistiques
    save_plots : bool
        Sauvegarder les graphiques
        
    Returns:
    --------
    None : Affiche les graphiques
    """
    # Configuration
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Analyse A/B Testing - Résultats Marketing', fontsize=16, fontweight='bold')
    
    # 1. Comparaison des taux de conversion
    ax1 = axes[0, 0]
    groups = ['Control\n(Old Page)', 'Treatment\n(New Page)']
    rates = [
        conversion_metrics['comparison']['control_rate'],
        conversion_metrics['comparison']['treatment_rate']
    ]
    colors = ['#ff7f7f', '#7fbf7f']
    
    bars = ax1.bar(groups, rates, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Taux de Conversion')
    ax1.set_title('Comparaison des Taux de Conversion')
    ax1.set_ylim(0, max(rates) * 1.2)
    
    # Ajout des valeurs sur les barres
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{rate:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Intervalles de confiance
    ax2 = axes[0, 1]
    ci_control = statistical_results['confidence_intervals_95']['control']
    ci_treatment = statistical_results['confidence_intervals_95']['treatment']
    
    ax2.errorbar([0], [rates[0]], yerr=[[rates[0] - ci_control[0]], [ci_control[1] - rates[0]]], 
                fmt='o', color='red', capsize=5, capthick=2, label='Control', markersize=8)
    ax2.errorbar([1], [rates[1]], yerr=[[rates[1] - ci_treatment[0]], [ci_treatment[1] - rates[1]]], 
                fmt='o', color='green', capsize=5, capthick=2, label='Treatment', markersize=8)
    
    ax2.set_xlim(-0.5, 1.5)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Control', 'Treatment'])
    ax2.set_ylabel('Taux de Conversion')
    ax2.set_title('Intervalles de Confiance 95%')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Distribution des conversions par jour
    ax3 = axes[1, 0]
    df['date'] = df['timestamp'].dt.date
    daily_conv = df.groupby(['date', 'group'])['converted'].mean().unstack()
    
    if 'control' in daily_conv.columns and 'treatment' in daily_conv.columns:
        daily_conv['control'].plot(ax=ax3, label='Control', alpha=0.7, color='red')
        daily_conv['treatment'].plot(ax=ax3, label='Treatment', alpha=0.7, color='green')
    
    ax3.set_ylabel('Taux de Conversion Quotidien')
    ax3.set_title('Évolution Temporelle des Conversions')
    ax3.legend()
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # 4. Résumé statistique
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Texte du résumé
    lift = conversion_metrics['comparison']['relative_lift_percent']
    p_value = statistical_results['z_test']['p_value']
    is_significant = statistical_results['z_test']['is_significant']
    
    summary_text = f"""
    RÉSULTATS A/B TESTING
    
    📊 Taux de Conversion:
    • Control: {rates[0]:.3f} ({rates[0]*100:.1f}%)
    • Treatment: {rates[1]:.3f} ({rates[1]*100:.1f}%)
    
    📈 Performance:
    • Lift Relatif: {lift:+.2f}%
    • Lift Absolu: {conversion_metrics['comparison']['absolute_lift']:+.4f}
    
    🔬 Test Statistique:
    • P-value: {p_value:.6f}
    • Significatif: {'✅ OUI' if is_significant else '❌ NON'}
    • Puissance: {statistical_results['statistical_power']['statistical_power']:.3f}
    
    💡 Recommandation:
    {'Adopter la nouvelle page' if is_significant and lift > 0 else 'Garder la page actuelle'}
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('ab_testing_results.png', dpi=300, bbox_inches='tight')
    
    plt.show()

def generate_business_recommendations(conversion_metrics, statistical_results, confidence_level=0.95):
    """
    Génère des recommandations business basées sur l'analyse
    
    Parameters:
    -----------
    conversion_metrics : dict
        Métriques de conversion
    statistical_results : dict
        Résultats statistiques
    confidence_level : float
        Niveau de confiance
        
    Returns:
    --------
    dict : Recommandations structurées
    """
    lift = conversion_metrics['comparison']['relative_lift_percent']
    is_significant = statistical_results['z_test']['is_significant']
    p_value = statistical_results['z_test']['p_value']
    power = statistical_results['statistical_power']['statistical_power']
    
    # Détermination de la recommandation principale
    if is_significant and lift > 0:
        main_recommendation = "ADOPTER LA NOUVELLE PAGE"
        confidence = "ÉLEVÉE"
        action = "Déployer immédiatement la nouvelle landing page"
    elif is_significant and lift < 0:
        main_recommendation = "GARDER L'ANCIENNE PAGE"
        confidence = "ÉLEVÉE"
        action = "Maintenir la page actuelle et investiguer pourquoi la nouvelle performe moins bien"
    elif not is_significant and abs(lift) < 2:
        main_recommendation = "RÉSULTAT NON CONCLUANT"
        confidence = "FAIBLE"
        action = "Prolonger le test ou accepter que les deux pages sont équivalentes"
    else:
        main_recommendation = "POURSUIVRE LE TEST"
        confidence = "MOYENNE"
        action = "Collecter plus de données pour atteindre la significativité statistique"
    
    # Calcul de l'impact business potentiel
    control_rate = conversion_metrics['comparison']['control_rate']
    treatment_rate = conversion_metrics['comparison']['treatment_rate']
    
    # Estimation sur 10,000 visiteurs mensuels
    monthly_visitors = 10000
    current_conversions = monthly_visitors * control_rate
    potential_conversions = monthly_visitors * treatment_rate
    additional_conversions = potential_conversions - current_conversions
    
    recommendations = {
        'executive_summary': {
            'recommendation': main_recommendation,
            'confidence_level': confidence,
            'statistical_significance': is_significant,
            'business_impact': f"{lift:+.2f}% de variation du taux de conversion"
        },
        'detailed_analysis': {
            'current_performance': {
                'control_conversion_rate': f"{control_rate:.3f} ({control_rate*100:.1f}%)",
                'sample_size_control': statistical_results['sample_sizes']['control']
            },
            'new_page_performance': {
                'treatment_conversion_rate': f"{treatment_rate:.3f} ({treatment_rate*100:.1f}%)",
                'sample_size_treatment': statistical_results['sample_sizes']['treatment']
            },
            'statistical_evidence': {
                'p_value': f"{p_value:.6f}",
                'is_significant_95': is_significant,
                'statistical_power': f"{power:.3f}",
                'confidence_interval_difference': statistical_results['confidence_intervals_95']['difference']
            }
        },
        'business_impact': {
            'relative_improvement': f"{lift:+.2f}%",
            'absolute_improvement': f"{conversion_metrics['comparison']['absolute_lift']:+.4f}",
            'estimated_monthly_impact': {
                'additional_conversions': f"{additional_conversions:+.0f} conversions/mois",
                'based_on_visitors': f"{monthly_visitors:,} visiteurs/mois"
            }
        },
        'action_plan': {
            'immediate_action': action,
            'next_steps': [
                "Documenter les résultats du test",
                "Communiquer les findings aux équipes marketing et produit",
                "Planifier le déploiement si recommandé",
                "Définir les métriques de suivi post-déploiement"
            ],
            'risks_and_considerations': [
                "Vérifier que les résultats sont cohérents sur différents segments",
                "S'assurer que l'amélioration se maintient dans le temps",
                "Considérer l'impact sur d'autres métriques (temps sur site, bounce rate, etc.)"
            ]
        }
    }
    
    return recommendations

def create_summary_report(df, conversion_metrics, statistical_results, recommendations):
    """
    Crée un rapport de synthèse complet
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset analysé
    conversion_metrics : dict
        Métriques de conversion
    statistical_results : dict
        Résultats statistiques
    recommendations : dict
        Recommandations business
        
    Returns:
    --------
    str : Rapport formaté
    """
    report = f"""
    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║                           RAPPORT A/B TESTING MARKETING                          ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝
    
    📅 Date d'analyse: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
    📊 Période des données: {df['timestamp'].min().strftime('%Y-%m-%d')} à {df['timestamp'].max().strftime('%Y-%m-%d')}
    
    ┌─ RÉSUMÉ EXÉCUTIF ─────────────────────────────────────────────────────────────────┐
    │ Recommandation: {recommendations['executive_summary']['recommendation']:<20} │
    │ Confiance: {recommendations['executive_summary']['confidence_level']:<25}      │
    │ Impact Business: {recommendations['executive_summary']['business_impact']:<15} │
    └───────────────────────────────────────────────────────────────────────────────────┘
    
    ┌─ MÉTRIQUES CLÉS ──────────────────────────────────────────────────────────────────┐
    │ Groupe Contrôle (Ancienne Page):                                                 │
    │   • Utilisateurs: {statistical_results['sample_sizes']['control']:,}                                          │
    │   • Conversions: {int(statistical_results['sample_sizes']['control'] * conversion_metrics['comparison']['control_rate']):,}                                         │
    │   • Taux: {conversion_metrics['comparison']['control_rate']:.3f} ({conversion_metrics['comparison']['control_rate']*100:.1f}%)                                   │
    │                                                                                   │
    │ Groupe Traitement (Nouvelle Page):                                               │
    │   • Utilisateurs: {statistical_results['sample_sizes']['treatment']:,}                                        │
    │   • Conversions: {int(statistical_results['sample_sizes']['treatment'] * conversion_metrics['comparison']['treatment_rate']):,}                                       │
    │   • Taux: {conversion_metrics['comparison']['treatment_rate']:.3f} ({conversion_metrics['comparison']['treatment_rate']*100:.1f}%)                                 │
    └───────────────────────────────────────────────────────────────────────────────────┘
    
    ┌─ ANALYSE STATISTIQUE ─────────────────────────────────────────────────────────────┐
    │ Test Z (Différence de proportions):                                              │
    │   • Statistique Z: {statistical_results['z_test']['z_statistic']:.4f}                                        │
    │   • P-value: {statistical_results['z_test']['p_value']:.6f}                                           │
    │   • Significatif (α=0.05): {'✅ OUI' if statistical_results['z_test']['is_significant'] else '❌ NON'}                                    │
    │                                                                                   │
    │ Puissance Statistique: {statistical_results['statistical_power']['statistical_power']:.3f}                                         │
    │ Intervalle de Confiance 95% (Différence): [{statistical_results['confidence_intervals_95']['difference'][0]:+.4f}, {statistical_results['confidence_intervals_95']['difference'][1]:+.4f}]     │
    └───────────────────────────────────────────────────────────────────────────────────┘
    
    ┌─ IMPACT BUSINESS ─────────────────────────────────────────────────────────────────┐
    │ Amélioration Relative: {conversion_metrics['comparison']['relative_lift_percent']:+.2f}%                                        │
    │ Amélioration Absolue: {conversion_metrics['comparison']['absolute_lift']:+.4f}                                      │
    │ Impact Mensuel Estimé: {recommendations['business_impact']['estimated_monthly_impact']['additional_conversions']}                              │
    └───────────────────────────────────────────────────────────────────────────────────┘
    
    ┌─ PLAN D'ACTION ───────────────────────────────────────────────────────────────────┐
    │ Action Immédiate:                                                                 │
    │ {recommendations['action_plan']['immediate_action']}                              │
    │                                                                                   │
    │ Prochaines Étapes:                                                                │
    """
    
    for step in recommendations['action_plan']['next_steps']:
        report += f"│ • {step:<75} │\n"
    
    report += """    └───────────────────────────────────────────────────────────────────────────────────┘
    """
    
    return report
