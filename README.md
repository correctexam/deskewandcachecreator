
# 🦀 Rust Deskew - Alignement de Scans (Implémentation Custom AKAZE)

Ce projet est une bibliothèque Rust pure permettant d'aligner (deskew) une image scannée par rapport à une image de référence.

Il a la particularité de ne **pas utiliser OpenCV** ni de bindings externes complexes. Il implémente son propre moteur de vision par ordinateur. Il offre 2 techniques d'alignement:
- une qui utilise une recherche de ronds noirs plein pour servir de point pour une transformation géométrique simple. 
- une inspirée d'**AKAZE** (Hessian Detector + LDB Descriptor) pour une robustesse maximale sur des documents, tout en restant léger et performant.

## ✨ Fonctionnalités Clés

*   **100% Rust** : Aucune installation de `libopencv` ou `cmake` requise.
*   **Algorithme Custom Robuste** :
    *   **Détection** : Utilise le **Déterminant de la Hessienne** (au lieu de FAST) pour trouver des points d'intérêt stables (blobs/coins) même avec du bruit de scan.
    *   **Description** : Implémente **LDB (Local Difference Binary)** sur 64 bits pour une comparaison ultra-rapide via la distance de Hamming.
*   **Optimisé pour les Scans** :
    *   Travaille uniquement sur les **4 coins (ROI)** pour éviter le bruit du texte central.
    *   Utilise une transformation de **Similitude** (Rotation + Translation + Échelle) pour éviter les déformations trapézoïdales irréalistes sur un scanner à plat.
*   **Haute Performance** : Parallélisation du traitement d'image (Warping) via **Rayon**.

## 🚀 Utilisation

TODO



## 📝 Licence

Ce projet est fourni à titre d'exemple éducatif et technique. Libre à vous de l'utiliser et de le modifier.


```bash
amqp-publish --url="amqp://rabbitmq:rabbitmq@localhost:5672"     --routing-key="scan_jobs_queue"     --body='{"pages_to_manage": "1-3", "exam_id": 855, "template_id": 857, "scan_id": 607, "algo": 2}'

curl -X POST http://localhost:8080/api/scansalign \
     -H "Content-Type: application/json" \
     -d '{
           "pages_to_manage": "1-3",
           "exam_id": 855,
           "template_id": 857,
           "scan_id": 607,
           "algo": 2,
           "heightresolution":2000,
           "corner_square_size":300,
           "min_radius":10.0
         }'
```