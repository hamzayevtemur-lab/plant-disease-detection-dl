import os
import time
from io import BytesIO
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from src.data.dataset import get_eval_transforms
from src.training.trainer import get_device, build_model

# 38 PlantVillage Disease Knowledge Base with Detailed Treatment Advice
CLASS_INFO = {
    0: {
        "label": "Apple___Apple_scab",
        "crop": "Apple",
        "disease": "Apple Scab",
        "is_healthy": False,
        "symptoms": "Olive-green to black velvety spots on leaves, causing premature defoliation and scarred fruit.",
        "organic_treatment": "Spray sulfur or copper-based fungicides in early spring. Rake and destroy fallen infected leaves.",
        "chemical_treatment": "Apply Captan, Mancozeb, or myclobutanil fungicides before rain events during bud break."
    },
    1: {
        "label": "Apple___Black_rot",
        "crop": "Apple",
        "disease": "Black Rot (Frog-Eye Leaf Spot)",
        "is_healthy": False,
        "symptoms": "Small purple specks that enlarge into brown circular lesions with purple borders.",
        "organic_treatment": "Prune dead wood and remove mummified apples from trees and ground.",
        "chemical_treatment": "Apply Captan or copper sulfate fungicide sprays during early spring growth."
    },
    2: {
        "label": "Apple___Cedar_apple_rust",
        "crop": "Apple",
        "disease": "Cedar Apple Rust",
        "is_healthy": False,
        "symptoms": "Bright orange-yellow spots on upper leaf surfaces; spore-producing cups underneath.",
        "organic_treatment": "Remove nearby eastern red cedar trees or prune rust galls from cedars in late winter.",
        "chemical_treatment": "Apply Immunox (myclobutanil) or Mancozeb at pink bud stage through petal fall."
    },
    3: {
        "label": "Apple___healthy",
        "crop": "Apple",
        "disease": "Healthy",
        "is_healthy": True,
        "symptoms": "Foliage is vibrant green, firm, and showing robust natural growth without lesions.",
        "organic_treatment": "Maintain balanced organic fertilization, proper canopy pruning, and deep watering.",
        "chemical_treatment": "No treatment required. Maintain standard preventive orchard management."
    },
    4: {
        "label": "Blueberry___healthy",
        "crop": "Blueberry",
        "disease": "Healthy",
        "is_healthy": True,
        "symptoms": "Leaves are glossy, vibrant green with healthy leaf margins and active shoot growth.",
        "organic_treatment": "Maintain acidic soil pH (4.5 - 5.5) with pine bark mulch and organic sulfur.",
        "chemical_treatment": "No treatment needed. Keep soil properly acidified and well drained."
    },
    5: {
        "label": "Cherry_(including_sour)___Powdery_mildew",
        "crop": "Cherry",
        "disease": "Powdery Mildew",
        "is_healthy": False,
        "symptoms": "White powdery fungal patches on new foliage and shoot tips, causing leaf curling.",
        "organic_treatment": "Apply neem oil, potassium bicarbonate, or sulfur spray during early shoot growth.",
        "chemical_treatment": "Use myclobutanil, propiconazole, or trifloxystrobin at petal fall."
    },
    6: {
        "label": "Cherry_(including_sour)___healthy",
        "crop": "Cherry",
        "disease": "Healthy",
        "is_healthy": True,
        "symptoms": "Leaves are dark green, smooth, and free of mildew or necrotic spots.",
        "organic_treatment": "Ensure good air circulation within tree canopy by seasonal pruning.",
        "chemical_treatment": "No treatment needed."
    },
    7: {
        "label": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "crop": "Corn (Maize)",
        "disease": "Gray Leaf Spot",
        "is_healthy": False,
        "symptoms": "Rectangular, tan to grayish lesions strictly bounded by leaf veins.",
        "organic_treatment": "Practice 2-3 year crop rotation and till crop residue after harvest.",
        "chemical_treatment": "Apply strobilurin (azoxystrobin) or triazole fungicides at tassel emergence."
    },
    8: {
        "label": "Corn_(maize)___Common_rust_",
        "crop": "Corn (Maize)",
        "disease": "Common Rust",
        "is_healthy": False,
        "symptoms": "Small, powdery golden-brown to cinnamon-colored pustules scattered on both leaf surfaces.",
        "organic_treatment": "Plant rust-resistant corn hybrids and maintain balanced soil potassium.",
        "chemical_treatment": "Apply pyraclostrobin or tebuconazole if rust appears before silking on susceptible hybrids."
    },
    9: {
        "label": "Corn_(maize)___Northern_Leaf_Blight",
        "crop": "Corn (Maize)",
        "disease": "Northern Leaf Blight",
        "is_healthy": False,
        "symptoms": "Long, cigar-shaped grayish-green to tan lesions (1 to 6 inches long).",
        "organic_treatment": "Use resistant hybrids and plow under corn stubble to reduce overwintering fungi.",
        "chemical_treatment": "Foliar fungicide application (Headline AMP, Quilt Xcel) when lesions reach ear leaf."
    },
    10: {
        "label": "Corn_(maize)___healthy",
        "crop": "Corn (Maize)",
        "disease": "Healthy",
        "is_healthy": True,
        "symptoms": "Broad, strong green leaves without spots, streaks, or pustules.",
        "organic_treatment": "Provide adequate nitrogen and steady drip irrigation during pollination.",
        "chemical_treatment": "No treatment required."
    },
    11: {
        "label": "Grape___Black_rot",
        "crop": "Grape",
        "disease": "Black Rot",
        "is_healthy": False,
        "symptoms": "Reddish-brown circular spots on leaves with dark borders and shriveled black berries.",
        "organic_treatment": "Remove and destroy mummies during pruning; apply liquid copper before rain.",
        "chemical_treatment": "Apply Mancozeb, Captan, or myclobutanil from early shoot growth until veraison."
    },
    12: {
        "label": "Grape___Esca_(Black_Measles)",
        "crop": "Grape",
        "disease": "Esca (Black Measles)",
        "is_healthy": False,
        "symptoms": "'Tiger-stripe' leaf patterns of yellow/brown interveinal necrosis.",
        "organic_treatment": "Prune out diseased wood during dry weather; treat pruning wounds with wound paste.",
        "chemical_treatment": "No direct cure; prevent wound infection with thiophanate-methyl pruning sealants."
    },
    13: {
        "label": "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
        "crop": "Grape",
        "disease": "Leaf Blight (Isariopsis)",
        "is_healthy": False,
        "symptoms": "Irregular dark brown patches on leaves causing premature yellowing and leaf drop.",
        "organic_treatment": "Improve canopy airflow with leaf thinning and apply copper hydroxide.",
        "chemical_treatment": "Spray azoxystrobin or copper oxychloride at early disease onset."
    },
    14: {
        "label": "Grape___healthy",
        "crop": "Grape",
        "disease": "Healthy",
        "is_healthy": True,
        "symptoms": "Vigorous green vine foliage with clear venation and healthy tendrils.",
        "organic_treatment": "Maintain balanced shoot positioning, soil aeration, and organic mulching.",
        "chemical_treatment": "No treatment required."
    },
    15: {
        "label": "Orange___Haunglongbing_(Citrus_greening)",
        "crop": "Orange / Citrus",
        "disease": "Citrus Greening (Huanglongbing)",
        "is_healthy": False,
        "symptoms": "Asymmetric yellow mottling on leaves, yellow shoots, and small lopsided bitter fruit.",
        "organic_treatment": "Control Asian citrus psyllids with horticultural oils and predatory wasps; remove infected trees.",
        "chemical_treatment": "Apply systemic insecticides (imidacloprid) to control the psyllid vector."
    },
    16: {
        "label": "Peach___Bacterial_spot",
        "crop": "Peach",
        "disease": "Bacterial Spot",
        "is_healthy": False,
        "symptoms": "Angular, water-soaked purple-black spots that drop out, giving a 'shot-hole' appearance.",
        "organic_treatment": "Apply dormant copper sprays and plant resistant peach cultivars.",
        "chemical_treatment": "Apply oxytetracycline or copper bactericide during early cover sprays."
    },
    17: {
        "label": "Peach___healthy",
        "crop": "Peach",
        "disease": "Healthy",
        "is_healthy": True,
        "symptoms": "Clean, elongated lanceolate leaves with deep green luster.",
        "organic_treatment": "Ensure well-drained soil and apply organic compost in early spring.",
        "chemical_treatment": "No treatment required."
    },
    18: {
        "label": "Pepper,_bell___Bacterial_spot",
        "crop": "Bell Pepper",
        "disease": "Bacterial Spot",
        "is_healthy": False,
        "symptoms": "Small, water-soaked circular lesions that turn dark brown with yellow halos.",
        "organic_treatment": "Use certified disease-free seed, avoid overhead watering, and apply copper soap.",
        "chemical_treatment": "Spray fixed copper mixed with Mancozeb on a 7-10 day preventive schedule."
    },
    19: {
        "label": "Pepper,_bell___healthy",
        "crop": "Bell Pepper",
        "disease": "Healthy",
        "is_healthy": True,
        "symptoms": "Smooth, glossy green leaves with strong central stems and active flower buds.",
        "organic_treatment": "Provide consistent moisture and calcium to prevent blossom end rot.",
        "chemical_treatment": "No treatment required."
    },
    20: {
        "label": "Potato___Early_blight",
        "crop": "Potato",
        "disease": "Early Blight",
        "is_healthy": False,
        "symptoms": "Dark brown to black spots with characteristic concentric rings (target-like pattern).",
        "organic_treatment": "Crop rotation away from nightshades; remove lower yellowing leaves; apply biofungicides.",
        "chemical_treatment": "Apply Chlorothalonil or Mancozeb starting when plants reach 6-8 inches tall."
    },
    21: {
        "label": "Potato___Late_blight",
        "crop": "Potato",
        "disease": "Late Blight (Phytophthora)",
        "is_healthy": False,
        "symptoms": "Large, dark water-soaked lesions with white fungal growth on undersides in humid weather.",
        "organic_treatment": "Destroy volunteer potatoes; use certified seed tubers; apply preventative copper.",
        "chemical_treatment": "Apply cymoxanil, mandipropamid, or fluopicolide fungicides immediately."
    },
    22: {
        "label": "Potato___healthy",
        "crop": "Potato",
        "disease": "Healthy",
        "is_healthy": True,
        "symptoms": "Robust, dark-green compound leaves free of blight lesions.",
        "organic_treatment": "Hill plants properly with straw/soil and maintain even soil moisture.",
        "chemical_treatment": "No treatment required."
    },
    23: {
        "label": "Raspberry___healthy",
        "crop": "Raspberry",
        "disease": "Healthy",
        "is_healthy": True,
        "symptoms": "Vibrant serrated leaves with clean silver-green undersides.",
        "organic_treatment": "Prune old floricanes after harvest and maintain open cane spacing.",
        "chemical_treatment": "No treatment required."
    },
    24: {
        "label": "Soybean___healthy",
        "crop": "Soybean",
        "disease": "Healthy",
        "is_healthy": True,
        "symptoms": "Trifoliate leaves are clean, uniformly green, and actively photosynthesizing.",
        "organic_treatment": "Maintain balanced soil phosphorus and inoculate seeds with Rhizobium.",
        "chemical_treatment": "No treatment required."
    },
    25: {
        "label": "Squash___Powdery_mildew",
        "crop": "Squash",
        "disease": "Powdery Mildew",
        "is_healthy": False,
        "symptoms": "Talcum-powder-like white spots expanding over upper and lower leaf surfaces.",
        "organic_treatment": "Spray baking soda solution (1 tbsp/gal water with horticultural oil) or milk spray (40/60).",
        "chemical_treatment": "Apply myclobutanil, triflumizole, or chlorothalonil at first sign of powdery spots."
    },
    26: {
        "label": "Strawberry___Leaf_scorch",
        "crop": "Strawberry",
        "disease": "Leaf Scorch",
        "is_healthy": False,
        "symptoms": "Irregular dark purple to brown blotches that coalesce and make the leaf look scorched/burned.",
        "organic_treatment": "Remove dead/infected leaves; avoid sprinkler irrigation; renovate beds after harvest.",
        "chemical_treatment": "Apply Captan or thiophanate-methyl fungicides during spring leaf emergence."
    },
    27: {
        "label": "Strawberry___healthy",
        "crop": "Strawberry",
        "disease": "Healthy",
        "is_healthy": True,
        "symptoms": "Rich green trifoliate leaves, firm stems, and healthy crowns.",
        "organic_treatment": "Keep beds cleanly mulched with pine straw to keep foliage dry.",
        "chemical_treatment": "No treatment required."
    },
    28: {
        "label": "Tomato___Bacterial_spot",
        "crop": "Tomato",
        "disease": "Bacterial Spot",
        "is_healthy": False,
        "symptoms": "Small, dark, greasy-looking angular spots with yellow halos on leaves.",
        "organic_treatment": "Apply copper soap fungicides; avoid working in plants while wet; practice crop rotation.",
        "chemical_treatment": "Spray copper hydroxide combined with Mancozeb weekly during humid conditions."
    },
    29: {
        "label": "Tomato___Early_blight",
        "crop": "Tomato",
        "disease": "Early Blight (Alternaria)",
        "is_healthy": False,
        "symptoms": "Brown to black spots with concentric rings surrounded by yellow tissue starting on lower leaves.",
        "organic_treatment": "Prune lower 12 inches of foliage; mulch soil heavily; apply Bacillus subtilis biofungicide.",
        "chemical_treatment": "Apply Chlorothalonil or azoxystrobin on a 7-14 day schedule."
    },
    30: {
        "label": "Tomato___Late_blight",
        "crop": "Tomato",
        "disease": "Late Blight",
        "is_healthy": False,
        "symptoms": "Large, dark water-soaked greasy patches on leaves and stems with white mold under humid conditions.",
        "organic_treatment": "Remove and bag infected plants immediately; plant resistant varieties like 'Mountain Magic'.",
        "chemical_treatment": "Apply Mandipropamid (Revus) or Chlorothalonil preventatively before rain."
    },
    31: {
        "label": "Tomato___Leaf_Mold",
        "crop": "Tomato",
        "disease": "Leaf Mold (Passalora fulva)",
        "is_healthy": False,
        "symptoms": "Pale yellow spots on upper leaf surfaces with olive-green velvety mold underneath.",
        "organic_treatment": "Increase greenhouse ventilation, reduce relative humidity below 85%, and apply copper.",
        "chemical_treatment": "Apply chlorothalonil, mancozeb, or cyazofamid fungicides."
    },
    32: {
        "label": "Tomato___Septoria_leaf_spot",
        "crop": "Tomato",
        "disease": "Septoria Leaf Spot",
        "is_healthy": False,
        "symptoms": "Numerous small circular spots with grayish-white centers and dark borders containing black specks.",
        "organic_treatment": "Mulch beds to stop soil splash, prune lower leaves, and apply copper fungicide.",
        "chemical_treatment": "Spray Chlorothalonil or Mancozeb starting at first flower cluster."
    },
    33: {
        "label": "Tomato___Spider_mites Two-spotted_spider_mite",
        "crop": "Tomato",
        "disease": "Two-Spotted Spider Mites",
        "is_healthy": False,
        "symptoms": "Fine yellow stippling on leaves, severe bronzing, and delicate silk webbing on leaf undersides.",
        "organic_treatment": "Release predatory mites (Phytoseiulus persimilis) or spray insecticidal soap / neem oil.",
        "chemical_treatment": "Apply selective miticides like abamectin or bifenazate."
    },
    34: {
        "label": "Tomato___Target_Spot",
        "crop": "Tomato",
        "disease": "Target Spot (Corynespora)",
        "is_healthy": False,
        "symptoms": "Small brown spots that enlarge into circular lesions with light brown centers and dark rings.",
        "organic_treatment": "Ensure good row spacing and airflow; destroy old crop residues.",
        "chemical_treatment": "Apply azoxystrobin, difenoconazole, or chlorothalonil."
    },
    35: {
        "label": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "crop": "Tomato",
        "disease": "Yellow Leaf Curl Virus (TYLCV)",
        "is_healthy": False,
        "symptoms": "Upward curling and yellowing of leaf margins, severe plant stunting, and bushy growth.",
        "organic_treatment": "Control silverleaf whitefly vector with insecticidal soap and yellow sticky traps; use row covers.",
        "chemical_treatment": "Apply systemic neonicotinoids or cyantraniliprole to manage whitefly populations."
    },
    36: {
        "label": "Tomato___Tomato_mosaic_virus",
        "crop": "Tomato",
        "disease": "Tomato Mosaic Virus (ToMV)",
        "is_healthy": False,
        "symptoms": "Mottled light and dark green patterns on leaves, leaf distortion, and 'fern-like' leaves.",
        "organic_treatment": "Disinfect tools with 20% non-fat milk solution; wash hands thoroughly; remove infected plants.",
        "chemical_treatment": "No chemical viricide exists. Plant virus-resistant varieties (labeled TMV/ToMV)."
    },
    37: {
        "label": "Tomato___healthy",
        "crop": "Tomato",
        "disease": "Healthy",
        "is_healthy": True,
        "symptoms": "Deep green foliage, robust branching, healthy blossoms, and no lesions.",
        "organic_treatment": "Stake plants upright, water at soil level, and apply organic tomato fertilizer.",
        "chemical_treatment": "No treatment required."
    }
}


def check_leaf_validity(image: Image.Image) -> Tuple[bool, str]:
    """
    Validates if an uploaded image contains a real botanical plant leaf vs out-of-distribution (UI screenshot, text slide, non-plant).
    """
    img_hsv = image.convert("HSV")
    hsv_arr = np.array(img_hsv)

    hue = hsv_arr[:, :, 0]
    sat = hsv_arr[:, :, 1]
    val = hsv_arr[:, :, 2]

    # Plant foliage pixels in PIL HSV (0-255 scale)
    # Green, yellow-green, yellowish, and natural plant browns
    plant_mask = ((hue >= 20) & (hue <= 105) & (sat >= 30) & (val >= 25))
    plant_ratio = float(np.mean(plant_mask))

    if plant_ratio < 0.15:
        return False, "No plant leaf detected in the image (0% plant foliage found). Please upload a clear photo of a crop leaf."

    return True, "Valid crop leaf."


class ModelManager:
    """Manages active deep learning model for real-time inference."""
    def __init__(self, experiments_dir: str = "experiments"):
        self.experiments_dir = experiments_dir
        self.device = get_device()
        self.active_model_name: Optional[str] = None
        self.active_model: Optional[torch.nn.Module] = None
        self.transform = get_eval_transforms()

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Returns list of all available trained models with their stats."""
        models = []
        if not os.path.exists(self.experiments_dir):
            return models

        for name in sorted(os.listdir(self.experiments_dir)):
            ckpt_path = os.path.join(self.experiments_dir, name, "best_model.pt")
            res_path = os.path.join(self.experiments_dir, name, "results.json")
            if os.path.isdir(os.path.join(self.experiments_dir, name)) and os.path.exists(ckpt_path):
                acc = "N/A"
                if os.path.exists(res_path):
                    try:
                        import json
                        with open(res_path) as f:
                            d = json.load(f)
                            acc = f"{d.get('test_accuracy', 0.0) * 100:.2f}%"
                    except Exception:
                        pass
                models.append({
                    "name": name,
                    "is_active": (name == self.active_model_name),
                    "test_accuracy": acc,
                    "checkpoint_exists": True
                })
        return models

    def load_model(self, model_name: Optional[str] = None) -> str:
        """Loads or switches the active model checkpoint."""
        if not model_name:
            priority = ["mobilenet_v3", "resnet18", "deeper_cnn_adamw", "deeper_cnn", "baseline"]
            available = [m["name"] for m in self.get_available_models()]
            for p in priority:
                if p in available:
                    model_name = p
                    break
            if not model_name and available:
                model_name = available[0]

        if not model_name:
            raise FileNotFoundError("No trained model checkpoints found in experiments/ directory.")

        ckpt_path = os.path.join(self.experiments_dir, model_name, "best_model.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found for model: {model_name}")

        print(f"Loading model '{model_name}' on device {self.device}...")
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        model_config = checkpoint["model_config"]

        model = build_model(model_config).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        self.active_model = model
        self.active_model_name = model_name
        print(f"Model '{model_name}' successfully loaded into memory!")
        return model_name

    def predict(self, image_bytes: bytes) -> Dict[str, Any]:
        """Runs inference on uploaded image bytes with Out-of-Distribution validation and top-5 probabilities."""
        if self.active_model is None:
            self.load_model()

        start_time = time.perf_counter()

        raw_image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Out-of-Distribution (OOD) leaf validation check
        is_leaf, validity_msg = check_leaf_validity(raw_image)

        if not is_leaf:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                "is_valid_leaf": False,
                "validation_warning": validity_msg,
                "model_used": self.active_model_name,
                "crop": "Non-Plant Image",
                "disease": "Invalid / Non-Leaf Input",
                "is_healthy": False,
                "confidence": 0.0,
                "symptoms": "The uploaded image is a document, UI screenshot, or non-botanical picture. Neural networks trained on crop leaves cannot diagnose non-plant objects.",
                "organic_treatment": "Please take or upload a clear, focused photo of a plant leaf.",
                "chemical_treatment": "Please take or upload a clear, focused photo of a plant leaf.",
                "top5": [],
                "latency_ms": round(elapsed_ms, 1)
            }

        tensor = self.transform(raw_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.active_model(tensor)
            probabilities = F.softmax(outputs, dim=1)[0]

        top5_prob, top5_indices = torch.topk(probabilities, 5)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        top_idx = int(top5_indices[0].item())
        top_confidence = float(top5_prob[0].item())

        info = CLASS_INFO.get(top_idx, {
            "label": f"Class_{top_idx}",
            "crop": "Unknown",
            "disease": f"Class {top_idx}",
            "is_healthy": False,
            "symptoms": "No symptoms documented.",
            "organic_treatment": "Consult an agricultural extension service.",
            "chemical_treatment": "Consult an agricultural extension service."
        })

        top5_list = []
        for prob, idx in zip(top5_prob, top5_indices):
            c_idx = int(idx.item())
            c_info = CLASS_INFO.get(c_idx, {})
            top5_list.append({
                "class_idx": c_idx,
                "crop": c_info.get("crop", "Unknown"),
                "disease": c_info.get("disease", f"Class {c_idx}"),
                "is_healthy": c_info.get("is_healthy", False),
                "confidence": round(float(prob.item()) * 100, 2),
                "raw_label": c_info.get("label", "")
            })

        return {
            "is_valid_leaf": True,
            "validation_warning": "",
            "model_used": self.active_model_name,
            "crop": info["crop"],
            "disease": info["disease"],
            "is_healthy": info["is_healthy"],
            "confidence": round(top_confidence * 100, 2),
            "symptoms": info["symptoms"],
            "organic_treatment": info["organic_treatment"],
            "chemical_treatment": info["chemical_treatment"],
            "top5": top5_list,
            "latency_ms": round(elapsed_ms, 1)
        }


# Global Singleton Manager
manager = ModelManager()


def get_model_manager() -> ModelManager:
    return manager
