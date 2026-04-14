import cv2

class RecipeInterface:
    def __init__(
        self,
        prompts,
        slider_up_gesture,
        slider_down_gesture,
        slider_neutral_gesture,
        on_recipe_change=None,
    ):
        self.recipe = {
            "Piano": 0.6,
            "Flute": 0.8,
            "Trumpet": 0.3
        }
        self.slider_up_gesture = slider_up_gesture
        self.slider_down_gesture = slider_down_gesture
        self.slider_neutral_gesture = slider_neutral_gesture
        self.on_recipe_change = on_recipe_change

        self.bar_positions = {k: (None,None) for k in self.recipe.keys()}
        self.bar_colors = {
            # note: color channels are flipped RGB -> BGR
            "Piano": (78, 166, 216)[::-1],
            "Flute": (250, 162, 75)[::-1],
            "Trumpet": (150, 187, 136)[::-1]
        }
    
    def draw_bars(self, webcam_frame, overlay_mask):
        height, width, num_channels = webcam_frame.shape
        num_bars = max(1, len(self.recipe))
        margin = 20
        self.bar_width = max(20, int((width - (num_bars + 1) * margin) / num_bars))
        bar_height = int(height * 0.7)
        self.bar_top = int(height * 0.1)
        self.bar_bottom = self.bar_top + bar_height

        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        font_thickness = 1

        positions = {}
        for i, (label, value) in enumerate(self.recipe.items()):
            x1 = margin + i * (self.bar_width + margin)
            x2 = x1 + self.bar_width

            bar_color = self.bar_colors.get(label, (88, 205, 54))

            cv2.rectangle(overlay_mask, (x1, self.bar_top), (x2, self.bar_bottom), bar_color, 2)

            v = max(0.0, min(1.0, float(value)))
            fill_height = int(bar_height * v)
            fill_top = self.bar_bottom - fill_height
            if fill_height > 0:
                cv2.rectangle(overlay_mask, (x1 + 2, fill_top), (x2 - 2, self.bar_bottom - 2), bar_color, -1)

            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            text_x = x1 + (self.bar_width - text_size[0]) // 2
            text_y = self.bar_bottom + text_size[1] + 10
            cv2.putText(overlay_mask, label, (text_x, text_y), font, font_scale, bar_color, font_thickness, cv2.LINE_AA)

            center_x = x1 + self.bar_width // 2
            center_y = fill_top if fill_height > 0 else self.bar_bottom
            positions[label] = (center_x, center_y)

        self.bar_positions = positions
        return positions
    
    def adjust_recipe(self, prompt, pointer_y):
        bar_height = max(1, self.bar_bottom - self.bar_top)
        clamped_y = max(self.bar_top, min(self.bar_bottom, int(pointer_y)))
        proportion = (self.bar_bottom - clamped_y) / bar_height
        proportion = max(0.0, min(1.0, float(proportion)))
        if prompt in self.recipe:
            previous = self.recipe[prompt]
            self.recipe[prompt] = proportion
            if previous != proportion:
                self.emit_recipe_update()

    def emit_recipe_update(self):
        """Send recipe update request whenever internal recipe changes."""
        if self.on_recipe_change is not None:
            self.on_recipe_change(dict(self.recipe))

    def update_positions(self, pointer_x, pointer_y, gesture, hand):
        closest_bar = None
        min_distance = float('inf')
        print(f"current gesture: {gesture}")
        
        for prompt, (bar_x, bar_y) in self.bar_positions.items():
            if bar_x is None:
                continue
            x_distance = abs(pointer_x - bar_x)
            if x_distance < self.bar_width and x_distance < min_distance:
                min_distance = x_distance
                closest_bar = prompt
        
        if closest_bar is not None:
            current_y = self.bar_positions[closest_bar][1]
            
            if gesture == self.slider_up_gesture and pointer_y < current_y:
                self.bar_positions[closest_bar] = (self.bar_positions[closest_bar][0], pointer_y)
                self.adjust_recipe(closest_bar, pointer_y)
            
            if gesture == self.slider_down_gesture and pointer_y > current_y:
                self.bar_positions[closest_bar] = (self.bar_positions[closest_bar][0], pointer_y)
                self.adjust_recipe(closest_bar, pointer_y)
    
    def change_prompts(self, new_prompt_list):
        self.recipe = {p: 0 for p in new_prompt_list}
        self.emit_recipe_update()

