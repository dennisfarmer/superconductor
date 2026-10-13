"""SuperConductor Atlas - entry point.

Server: the harmony graph, the beam-search walker, and the session API.
Client: the atlas UI (frontend.cl.jac).
"""

import from services.atlas {
    get_state, set_key, set_tempo, set_style_layers, set_texture, set_llm, explore,
    add_path, undo, clear_progression, describe, add_chords, restart_engine, set_loop, add_option,
    commit_progression, delete_saved, resume_saved, swap_step, shutdown_app, set_drums
}

cl {
    import "./assets/atlas.css";
    import from .frontend { app as AtlasApp }

    """The atlas; it shows the Harmony or the Drums page (/#drums) under one top bar."""
    def:pub app -> JsxElement {
        return <AtlasApp/>;
    }
}
