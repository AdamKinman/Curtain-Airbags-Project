"""
Generate car designs using Stable Diffusion.

This module provides functions to generate car design images using Stable Diffusion XL,
allowing users to specify car types and descriptions, or use default alternatives.
"""

import os
from typing import Optional, List, Literal
from enum import Enum
from dataclasses import dataclass, field

import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

from config import GENERATED_IMAGES_FOLDER


# --- Default Car Types and Descriptions ---

class CarType(Enum):
    """Predefined car types for design generation."""
    COMPACT_HATCHBACK = "compact_hatchback"
    SEDAN = "sedan"
    SUV = "suv"
    COUPE = "coupe"
    WAGON = "wagon"
    CROSSOVER = "crossover"
    SPORTS_CAR = "sports_car"
    MINIVAN = "minivan"
    PICKUP_TRUCK = "pickup_truck"
    LUXURY_SEDAN = "luxury_sedan"


@dataclass
class CarDescription:
    """Describes characteristics of a car for image generation."""
    car_type: str
    brand_style: Optional[str] = None
    color: Optional[str] = None
    era: Optional[str] = None  # e.g., "modern", "vintage", "futuristic"
    features: List[str] = field(default_factory=list)
    custom_details: Optional[str] = None


# --- Default Descriptions for Each Car Type ---

DEFAULT_DESCRIPTIONS = {
    CarType.COMPACT_HATCHBACK: CarDescription(
        car_type="compact hatchback",
        brand_style=None,
        color="silver",
        era="modern",
        features=[]
    ),
    CarType.SEDAN: CarDescription(
        car_type="sedan",
        brand_style=None,
        color="blue",
        era="modern",
        features=[]
    ),
    CarType.SUV: CarDescription(
        car_type="SUV",
        brand_style=None,
        color="black",
        era="modern",
        features=[]
    ),
    CarType.COUPE: CarDescription(
        car_type="sports coupe",
        brand_style=None,
        color="red",
        era="modern",
        features=[]
    ),
    CarType.WAGON: CarDescription(
        car_type="station wagon",
        brand_style=None,
        color="white",
        era="modern",
        features=[]
    ),
    CarType.CROSSOVER: CarDescription(
        car_type="crossover",
        brand_style=None,
        color="orange",
        era="modern",
        features=[]
    ),
    CarType.SPORTS_CAR: CarDescription(
        car_type="sports car",
        brand_style=None,
        color="yellow",
        era="modern",
        features=[]
    ),
    CarType.MINIVAN: CarDescription(
        car_type="minivan",
        brand_style=None,
        color="gold",
        era="modern",
        features=[]
    ),
    CarType.PICKUP_TRUCK: CarDescription(
        car_type="pickup truck",
        brand_style=None,
        color="dark blue",
        era="modern",
        features=[]
    ),
    CarType.LUXURY_SEDAN: CarDescription(
        car_type="luxury sedan",
        brand_style=None,
        color="black",
        era="modern",
        features=[]
    ),
}


# --- Prompt Building ---

def build_prompt(description: CarDescription) -> str:
    """
    Build a Stable Diffusion prompt from a car description.
    
    Args:
        description: A CarDescription object with car characteristics.
        
    Returns:
        A detailed prompt string for image generation.
    """
    # Build the main subject description
    subject_parts = []
    
    if description.color:
        subject_parts.append(description.color)
    
    if description.era and description.era != "modern":
        era_adjectives = {
            "vintage": "vintage 1960s",
            "futuristic": "futuristic concept",
            "retro": "retro-styled",
        }
        subject_parts.append(era_adjectives.get(description.era, ""))
    
    subject_parts.append(description.car_type)
    
    if description.brand_style:
        subject_parts.append(f"with {description.brand_style}")
    
    subject = " ".join(filter(None, subject_parts))
    
    # Add features if present
    features_str = ""
    if description.features:
        features_str = f", featuring {', '.join(description.features)}"
    
    # Add custom details
    custom_str = ""
    if description.custom_details:
        custom_str = f", {description.custom_details}"
    
    # Build a single coherent prompt - emphasize SINGLE car and strict side view
    prompt = (
        f"A single {subject} car{features_str}{custom_str}. "
        f"Perfect 90-degree side profile view, lateral view showing entire left side of vehicle, "
        f"wheels pointing left, perfectly horizontal orientation, no rotation, no angle, "
        f"orthographic side elevation, car facing left, "
        f"professional automotive photography, one complete vehicle centered in frame, "
        f"studio lighting, clean neutral gray gradient background, photorealistic, "
        f"8k resolution"
    )
    
    return prompt


DEFAULT_NEGATIVE_PROMPT = (
    "multiple cars, two cars, many cars, several vehicles, car lot, parking lot, "
    "traffic, fleet, showroom with multiple vehicles, collage, split image, "
    "fragmented, disconnected parts, floating parts, dismembered, broken, "
    "incomplete car, partial car, cropped car, cut off, "
    "deformed, distorted, warped, twisted, melted, corrupted, glitched, "
    "extra wheels, extra doors, wrong proportions, asymmetric, "
    "mutated, malformed, disfigured, bad anatomy, "
    "blurry, low quality, pixelated, jpeg artifacts, noise, grainy, "
    "cartoon, anime, illustration, drawing, sketch, painting, CGI, 3D render, "
    "text, watermark, logo, signature, border, frame, "
    "background cars, cars in distance, reflection of other cars, "
    "front view, rear view, three-quarter view, angled view, diagonal view, "
    "tilted, rotated, turned, perspective view, birds eye view, top view, "
    "looking down, looking up, fisheye, wide angle distortion"
)


# --- Image Generation Pipeline ---

class CarDesignGenerator:
    """
    A class to generate car design images using Stable Diffusion XL.
    """
    
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        device: Optional[str] = None,
        use_fp16: bool = True,
    ):
        """
        Initialize the generator with the Stable Diffusion model.
        
        Args:
            model_id: The Hugging Face model ID for Stable Diffusion.
            device: The device to run on ("cuda", "cpu", or None for auto-detect).
            use_fp16: Whether to use half-precision for memory efficiency.
        """
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16 and self.device == "cuda"
        self.pipe = None
        
    def load_model(self) -> None:
        """Load the Stable Diffusion model into memory."""
        if self.pipe is not None:
            return
            
        print(f"Loading model {self.model_id} on {self.device}...")
        
        dtype = torch.float16 if self.use_fp16 else torch.float32
        
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
        ).to(self.device)
        
        print("Model loaded successfully.")
    
    def generate(
        self,
        description: Optional[CarDescription] = None,
        car_type: Optional[CarType] = None,
        custom_prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        output_path: Optional[str] = None,
        filename: Optional[str] = None,
        guidance_scale: float = 9.0,
        num_inference_steps: int = 50,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Generate a single car design image.
        
        Args:
            description: A CarDescription object for the car. If None, uses car_type default.
            car_type: A CarType enum value to use default description. Ignored if description is provided.
            custom_prompt: Override the auto-generated prompt with a custom one.
            negative_prompt: Custom negative prompt. Uses default if None.
            output_path: Directory to save the image. Defaults to GENERATED_IMAGES_FOLDER.
            filename: Name for the output file. Auto-generated if None.
            guidance_scale: Classifier-free guidance scale (higher = more prompt adherence).
            num_inference_steps: Number of denoising steps.
            seed: Random seed for reproducibility. None for random.
            
        Returns:
            The generated PIL Image object.
        """
        self.load_model()
        
        # Determine the prompt
        if custom_prompt:
            prompt = custom_prompt
        elif description:
            prompt = build_prompt(description)
        elif car_type:
            prompt = build_prompt(DEFAULT_DESCRIPTIONS[car_type])
        else:
            # Default to sedan if nothing specified
            prompt = build_prompt(DEFAULT_DESCRIPTIONS[CarType.SEDAN])
        
        # Use default negative prompt if not provided
        neg_prompt = negative_prompt or DEFAULT_NEGATIVE_PROMPT
        
        # Set seed for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        print("Generating image...")
        result = self.pipe(
            prompt=prompt,
            negative_prompt=neg_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        
        image = result.images[0]
        
        # Save the image
        output_dir = output_path or GENERATED_IMAGES_FOLDER
        os.makedirs(output_dir, exist_ok=True)
        
        if filename is None:
            # Generate a unique filename by finding the next available index
            existing_files = os.listdir(output_dir)
            existing_indices = []
            for f in existing_files:
                if f.startswith("car_design_") and f.endswith(".png"):
                    try:
                        # Extract the number from filenames like "car_design_0001.png"
                        num_str = f[len("car_design_"):-len(".png")]
                        existing_indices.append(int(num_str))
                    except ValueError:
                        pass
            index = max(existing_indices, default=0) + 1
            filename = f"car_design_{index:04d}.png"
        
        full_path = os.path.join(output_dir, filename)
        image.save(full_path)
        print(f"Image saved to: {full_path}")
        
        return image
    
    def generate_batch(
        self,
        descriptions: Optional[List[CarDescription]] = None,
        car_types: Optional[List[CarType]] = None,
        count: int = 1,
        output_path: Optional[str] = None,
        filename_prefix: str = "car_design",
        guidance_scale: float = 9.0,
        num_inference_steps: int = 50,
        seeds: Optional[List[int]] = None,
    ) -> List[Image.Image]:
        """
        Generate multiple car design images.
        
        Args:
            descriptions: List of CarDescription objects. Takes priority over car_types.
            car_types: List of CarType enums to generate. Used if descriptions is None.
            count: Number of images to generate (only used if both descriptions and car_types are None).
            output_path: Directory to save images. Defaults to GENERATED_IMAGES_FOLDER.
            filename_prefix: Prefix for output filenames.
            guidance_scale: Classifier-free guidance scale.
            num_inference_steps: Number of denoising steps.
            seeds: List of seeds for reproducibility. None for random.
            
        Returns:
            List of generated PIL Image objects.
        """
        self.load_model()
        
        # Determine what to generate
        if descriptions:
            items = descriptions
            use_descriptions = True
        elif car_types:
            items = car_types
            use_descriptions = False
        else:
            # Generate 'count' images with random default types
            all_types = list(CarType)
            items = [all_types[i % len(all_types)] for i in range(count)]
            use_descriptions = False
        
        images = []
        for i, item in enumerate(items):
            print(f"\nGenerating image {i + 1}/{len(items)}...")
            
            seed = seeds[i] if seeds and i < len(seeds) else None
            filename = f"{filename_prefix}_{i + 1:04d}.png"
            
            if use_descriptions:
                image = self.generate(
                    description=item,
                    output_path=output_path,
                    filename=filename,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    seed=seed,
                )
            else:
                image = self.generate(
                    car_type=item,
                    output_path=output_path,
                    filename=filename,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    seed=seed,
                )
            
            images.append(image)
        
        print(f"\nGeneration complete. {len(images)} images created.")
        return images


# --- Convenience Functions ---

def generate_car_design(
    car_type: Optional[str] = None,
    color: Optional[str] = None,
    brand_style: Optional[str] = None,
    era: str = "modern",
    features: Optional[List[str]] = None,
    custom_details: Optional[str] = None,
    output_path: Optional[str] = None,
    filename: Optional[str] = None,
    seed: Optional[int] = None,
) -> Image.Image:
    """
    Convenience function to generate a car design image.
    
    Args:
        car_type: Type of car (e.g., "sedan", "SUV", "coupe"). Defaults to "sedan".
        color: Car color (e.g., "metallic blue", "racing red").
        brand_style: Style description (e.g., "German luxury", "Japanese efficiency").
        era: Design era ("modern", "vintage", "futuristic", "retro").
        features: List of design features.
        custom_details: Additional custom details for the prompt.
        output_path: Directory to save the image. Defaults to GENERATED_IMAGES_FOLDER.
        filename: Output filename. Auto-generated if None.
        seed: Random seed for reproducibility.
        
    Returns:
        The generated PIL Image object.
    """
    description = CarDescription(
        car_type=car_type or "sedan",
        color=color,
        brand_style=brand_style,
        era=era,
        features=features or [],
        custom_details=custom_details,
    )
    
    generator = CarDesignGenerator()
    return generator.generate(
        description=description,
        output_path=output_path,
        filename=filename,
        seed=seed,
    )


def generate_car_designs_from_type(
    car_type: CarType,
    count: int = 1,
    output_path: Optional[str] = None,
    seeds: Optional[List[int]] = None,
) -> List[Image.Image]:
    """
    Generate multiple car designs from a predefined car type.
    
    Args:
        car_type: A CarType enum value.
        count: Number of images to generate.
        output_path: Directory to save images. Defaults to GENERATED_IMAGES_FOLDER.
        seeds: List of seeds for reproducibility.
        
    Returns:
        List of generated PIL Image objects.
    """
    generator = CarDesignGenerator()
    return generator.generate_batch(
        car_types=[car_type] * count,
        output_path=output_path,
        seeds=seeds,
    )


def list_available_car_types() -> List[str]:
    """Return a list of available predefined car types."""
    return [ct.value for ct in CarType]


def get_default_description(car_type: CarType) -> CarDescription:
    """Get the default description for a car type."""
    return DEFAULT_DESCRIPTIONS.get(car_type)


# --- Main (Example Usage) ---

if __name__ == "__main__":
    print("Available car types:")
    for ct in CarType:
        print(f"  - {ct.value}")
    
    print("\n--- Example: Generate a single car design ---")
    
    # Example 1: Use a predefined car type
    image = generate_car_designs_from_type(CarType.SPORTS_CAR, count=1)
    
    # Example 2: Custom description
    # custom_desc = CarDescription(
    #     car_type="electric crossover",
    #     brand_style="Tesla-inspired minimalist",
    #     color="midnight silver metallic",
    #     era="futuristic",
    #     features=["flush door handles", "panoramic glass roof", "LED light bar"],
    #     custom_details="Sleek aerodynamic profile with minimal panel gaps."
    # )
    # generator = CarDesignGenerator()
    # image = generator.generate(description=custom_desc)
    
    # Example 3: Quick generation with convenience function
    # image = generate_car_design(
    #     car_type="compact SUV",
    #     color="forest green",
    #     brand_style="Scandinavian minimalist",
    #     features=["panoramic roof", "LED headlights", "alloy wheels"],
    #     seed=42,
    # )
    
    print("\nTo generate images, uncomment one of the examples above and run the script.")
    print(f"Default output location: {GENERATED_IMAGES_FOLDER}")
