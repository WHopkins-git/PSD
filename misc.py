y_true_bin = y_true[in_bin]
        y_pred_proba_bin = y_pred_proba[in_bin]
        
        # ROC for this energy bin
        fpr_bin, tpr_bin, thresh_bin = roc_curve(y_true_bin, y_pred_proba_bin)
        auc_bin = auc(fpr_bin, tpr_bin)
        
        # Operating point
        target_idx_bin = np.argmin(np.abs(tpr_bin - neutron_acceptance))
        gamma_misid_bin = fpr_bin[target_idx_bin]
        
        # PSD Figure of Merit (FoM)
        neutrons_bin = y_pred_proba_bin[y_true_bin == 1]
        gammas_bin = y_pred_proba_bin[y_true_bin == 0]
        
        if len(neutrons_bin) > 10 and len(gammas_bin) > 10:
            mean_n = neutrons_bin.mean()
            mean_g = gammas_bin.mean()
            fwhm_n = 2.355 * neutrons_bin.std()
            fwhm_g = 2.355 * gammas_bin.std()
            
            if (fwhm_n + fwhm_g) > 0:
                fom = abs(mean_n - mean_g) / (fwhm_n + fwhm_g)
            else:
                fom = 0
        else:
            fom = 0
        
        results['by_energy'][bin_name] = {
            'e_min': e_min,
            'e_max': e_max,
            'n_events': in_bin.sum(),
            'n_neutrons': (y_true_bin == 1).sum(),
            'n_gammas': (y_true_bin == 0).sum(),
            'roc_auc': auc_bin,
            'gamma_misid_rate': gamma_misid_bin,
            'fom': fom,
            'fpr': fpr_bin,
            'tpr': tpr_bin
        }
        
        print(f"\n{bin_name}:")
        print(f"  Events: {in_bin.sum()} (n={results['by_energy'][bin_name]['n_neutrons']}, "
              f"γ={results['by_energy'][bin_name]['n_gammas']})")
        print(f"  ROC AUC: {auc_bin:.4f}")
        print(f"  γ mis-ID @ {neutron_acceptance*100:.1f}% n: {gamma_misid_bin*100:.3f}%")
        print(f"  FoM: {fom:.3f}")
    
    # =========================================================================
    # CONFUSION MATRIX AT OPERATING POINT
    # =========================================================================
    
    y_pred = (y_pred_proba >= threshold_at_target).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    results['overall']['confusion_matrix'] = cm
    
    print("\n" + "="*70)
    print("CONFUSION MATRIX (at operating point)")
    print("="*70)
    print("                 Predicted")
    print("                 Gamma  Neutron")
    print(f"Actual Gamma     {cm[0,0]:5d}  {cm[0,1]:5d}   (γ mis-ID: {cm[0,1]/(cm[0,0]+cm[0,1])*100:.2f}%)")
    print(f"Actual Neutron   {cm[1,0]:5d}  {cm[1,1]:5d}   (n accept: {cm[1,1]/(cm[1,0]+cm[1,1])*100:.2f}%)")
    
    return results


def plot_evaluation_results(results, save_path='psd_evaluation.png'):
    """
    Visualize evaluation results
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # -------------------------------------------------------------------------
    # 1. Overall ROC curve
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    
    fpr = results['overall']['fpr']
    tpr = results['overall']['tpr']
    auc_val = results['overall']['roc_auc']
    
    ax1.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {auc_val:.4f}')
    ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random')
    
    # Mark operating point
    gamma_misid = results['overall']['gamma_misid_rate']
    n_accept = results['overall']['neutron_acceptance']
    ax1.plot(gamma_misid, n_accept, 'go', markersize=12, 
            label=f'Operating point\n({gamma_misid*100:.3f}% γ mis-ID)')
    
    ax1.set_xlabel('Gamma Mis-ID Rate (FPR)', fontsize=12)
    ax1.set_ylabel('Neutron Acceptance (TPR)', fontsize=12)
    ax1.set_title('ROC Curve - Overall', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # -------------------------------------------------------------------------
    # 2. ROC curves by energy bin
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(results['by_energy'])))
    
    for (bin_name, data), color in zip(results['by_energy'].items(), colors):
        ax2.plot(data['fpr'], data['tpr'], linewidth=2, color=color,
                label=f"{bin_name.replace('_keVee', '')} (AUC={data['roc_auc']:.3f})")
    
    ax2.plot([0, 1], [0, 1], 'r--', linewidth=2, alpha=0.5)
    ax2.set_xlabel('Gamma Mis-ID Rate', fontsize=12)
    ax2.set_ylabel('Neutron Acceptance', fontsize=12)
    ax2.set_title('ROC Curves by Energy Bin', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # 3. Gamma mis-ID rate vs energy
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    
    bin_names = []
    gamma_misids = []
    e_centers = []
    
    for bin_name, data in results['by_energy'].items():
        bin_names.append(bin_name.replace('_keVee', ''))
        gamma_misids.append(data['gamma_misid_rate'] * 100)
        e_centers.append((data['e_min'] + data['e_max']) / 2)
    
    ax3.plot(e_centers, gamma_misids, 'bo-', linewidth=2, markersize=8)
    ax3.set_xlabel('Energy (keVee)', fontsize=12)
    ax3.set_ylabel('Gamma Mis-ID Rate (%)', fontsize=12)
    ax3.set_title(f'γ Mis-ID vs Energy\n(at {results["overall"]["neutron_acceptance"]*100:.0f}% n acceptance)', 
                 fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # -------------------------------------------------------------------------
    # 4. FoM vs energy
    # -------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 0])
    
    foms = [data['fom'] for data in results['by_energy'].values()]
    
    ax4.plot(e_centers, foms, 'ro-', linewidth=2, markersize=8)
    ax4.axhline(1.0, color='green', linestyle='--', linewidth=2, label='FoM = 1.0 (good)')
    ax4.axhline(1.5, color='orange', linestyle='--', linewidth=2, label='FoM = 1.5 (excellent)')
    ax4.set_xlabel('Energy (keVee)', fontsize=12)
    ax4.set_ylabel('Figure of Merit', fontsize=12)
    ax4.set_title('PSD Figure of Merit vs Energy', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # 5. AUC vs energy
    # -------------------------------------------------------------------------
    ax5 = fig.add_subplot(gs[1, 1])
    
    aucs = [data['roc_auc'] for data in results['by_energy'].values()]
    
    ax5.plot(e_centers, aucs, 'mo-', linewidth=2, markersize=8)
    ax5.axhline(0.99, color='green', linestyle='--', linewidth=2, alpha=0.5)
    ax5.set_xlabel('Energy (keVee)', fontsize=12)
    ax5.set_ylabel('ROC AUC', fontsize=12)
    ax5.set_title('ROC AUC vs Energy', fontsize=14, fontweight='bold')
    ax5.set_ylim([0.9, 1.0])
    ax5.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # 6. Event distribution by energy
    # -------------------------------------------------------------------------
    ax6 = fig.add_subplot(gs[1, 2])
    
    n_neutrons = [data['n_neutrons'] for data in results['by_energy'].values()]
    n_gammas = [data['n_gammas'] for data in results['by_energy'].values()]
    
    x = np.arange(len(bin_names))
    width = 0.35
    
    ax6.bar(x - width/2, n_neutrons, width, label='Neutrons', color='red', alpha=0.7)
    ax6.bar(x + width/2, n_gammas, width, label='Gammas', color='blue', alpha=0.7)
    
    ax6.set_xlabel('Energy Bin', fontsize=12)
    ax6.set_ylabel('Event Count', fontsize=12)
    ax6.set_title('Event Distribution', fontsize=14, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(bin_names, rotation=45, ha='right')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.set_yscale('log')
    
    # -------------------------------------------------------------------------
    # 7. Confusion matrix heatmap
    # -------------------------------------------------------------------------
    ax7 = fig.add_subplot(gs[2, 0])
    
    cm = results['overall']['confusion_matrix']
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    im = ax7.imshow(cm_norm, cmap='Blues', aspect='auto')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax7.text(j, i, f'{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)',
                          ha="center", va="center", color="white" if cm_norm[i, j] > 0.5 else "black",
                          fontsize=12, fontweight='bold')
    
    ax7.set_xticks([0, 1])
    ax7.set_yticks([0, 1])
    ax7.set_xticklabels(['Gamma', 'Neutron'], fontsize=12)
    ax7.set_yticklabels(['Gamma', 'Neutron'], fontsize=12)
    ax7.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax7.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax7, label='Normalized Rate')
    
    # -------------------------------------------------------------------------
    # 8. Operating point tradeoff curve
    # -------------------------------------------------------------------------
    ax8 = fig.add_subplot(gs[2, 1])
    
    # Scan thresholds to show tradeoff
    thresholds_scan = np.linspace(0, 1, 100)
    
    ax8.plot(fpr * 100, tpr * 100, 'b-', linewidth=2)
    ax8.plot(gamma_misid * 100, n_accept * 100, 'go', markersize=15,
            label=f'Operating Point\n({gamma_misid*100:.3f}% γ mis-ID)')
    
    ax8.set_xlabel('Gamma Mis-ID Rate (%)', fontsize=12)
    ax8.set_ylabel('Neutron Acceptance (%)', fontsize=12)
    ax8.set_title('Operating Point Tradeoff', fontsize=14, fontweight='bold')
    ax8.legend(fontsize=10)
    ax8.grid(True, alpha=0.3)
    ax8.set_xlim([0, 10])  # Focus on low mis-ID region
    ax8.set_ylim([90, 100])
    
    # -------------------------------------------------------------------------
    # 9. Summary statistics table
    # -------------------------------------------------------------------------
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    summary_text = f"""
    PERFORMANCE SUMMARY
    {'='*35}
    
    Overall:
      ROC AUC: {results['overall']['roc_auc']:.4f}
      Neutron Acceptance: {n_accept*100:.2f}%
      Gamma Mis-ID: {gamma_misid*100:.3f}%
    
    By Energy:
    """
    
    for bin_name, data in results['by_energy'].items():
        summary_text += f"""
      {bin_name.replace('_keVee', '')}:
        FoM: {data['fom']:.3f}
        γ mis-ID: {data['gamma_misid_rate']*100:.3f}%
        AUC: {data['roc_auc']:.4f}
    """
    
    ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('PSD Classification Performance Evaluation', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Evaluation plots saved to {save_path}")
    
    return fig


def compare_methods(results_dict, save_path='method_comparison.png'):
    """
    Compare multiple methods side-by-side
    
    Parameters:
    -----------
    results_dict : dict
        {method_name: evaluation_results}
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
    
    # ROC curves
    ax = axes[0, 0]
    for (method, results), color in zip(results_dict.items(), colors):
        fpr = results['overall']['fpr']
        tpr = results['overall']['tpr']
        auc_val = results['overall']['roc_auc']
        ax.plot(fpr, tpr, linewidth=2, color=color, label=f'{method} (AUC={auc_val:.4f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.3)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Gamma mis-ID rates
    ax = axes[0, 1]
    methods = list(results_dict.keys())
    gamma_misids = [r['overall']['gamma_misid_rate']*100 for r in results_dict.values()]
    
    bars = ax.bar(methods, gamma_misids, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Gamma Mis-ID Rate (%)', fontsize=12)
    ax.set_title('Gamma Mis-ID Comparison\n(at 99% neutron acceptance)', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, gamma_misids):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.3f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # FoM comparison
    ax = axes[1, 0]
    
    for (method, results), color in zip(results_dict.items(), colors):
        e_centers = []
        foms = []
        for bin_name, data in results['by_energy'].items():
            e_centers.append((data['e_min'] + data['e_max']) / 2)
            foms.append(data['fom'])
        
        ax.plot(e_centers, foms, 'o-', linewidth=2, markersize=8, 
               color=color, label=method)
    
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=2, alpha=0.5)
    ax.set_xlabel('Energy (keVee)', fontsize=12)
    ax.set_ylabel('Figure of Merit', fontsize=12)
    ax.set_title('FoM vs Energy', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Summary table
    ax = axes[1, 1]
    ax.axis('off')
    
    # Create comparison table
    table_data = []
    table_data.append(['Method', 'AUC', 'γ mis-ID (%)', 'Avg FoM'])
    
    for method, results in results_dict.items():
        auc_val = results['overall']['roc_auc']
        gamma_misid = results['overall']['gamma_misid_rate'] * 100
        
        foms = [data['fom'] for data in results['by_energy'].values()]
        avg_fom = np.mean(foms) if foms else 0
        
        table_data.append([
            method,
            f'{auc_val:.4f}',
            f'{gamma_misid:.3f}',
            f'{avg_fom:.3f}'
        ])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        color = '#f0f0f0' if i % 2 == 0 else 'white'
        for j in range(4):
            table[(i, j)].set_facecolor(color)
    
    ax.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.suptitle('Method Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    print(f"\n✓ Comparison plots saved to {save_path}")
    
    return fig


# Example usage
if __name__ == "__main__":
    print("Proper PSD Evaluation Metrics")
    print("="*70)
    print("\nKey improvements over simple accuracy:")
    print("  ✓ Gamma mis-ID rate at fixed neutron acceptance")
    print("  ✓ Energy-dependent performance (FoM, AUC)")
    print("  ✓ ROC curves per energy bin")
    print("  ✓ Operating point analysis")
    print("  ✓ Proper reporting for class imbalance")
    print("\nUsage:")
    print("  results = evaluate_psd_classifier(y_true, y_pred_proba, energies)")
    print("  plot_evaluation_results(results)")