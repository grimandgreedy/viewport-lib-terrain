//! How the renderer reads a frame's [`TerrainCollection`]. Needs the `gpu`
//! feature.

use std::any::Any;

use viewport_lib::ItemSettings;
use viewport_lib::plugin_api::PluginItemCollection;
use viewport_lib::renderer::PickId;

use crate::item::TerrainCollection;

impl PluginItemCollection for TerrainCollection {
    fn len(&self) -> usize {
        self.items.len()
    }

    fn item_settings(&self, index: usize) -> &ItemSettings {
        &self.items[index].settings
    }

    fn pick_id(&self, index: usize) -> PickId {
        self.items[index].settings.pick_id
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
