import os
import streamlit as st
import streamlit.components.v1 as components

# Ambil direktori absolut dari keyup_component
parent_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(parent_dir, "keyup_component")

# Deklarasikan Streamlit Component lokal
_component_func = components.declare_component("keyup_numeric", path=build_dir)


def keyup_numeric(
    label: str,
    value: str = "",
    key: str | None = None,
    placeholder: str = "",
    disabled: bool = False,
):
    """
    Component custom Streamlit yang mendeteksi perubahan input secara real-time (keyup)
    dan memblokir huruf secara fisik (hanya menerima angka dan 1 pemisah desimal).
    """
    if key is not None and key not in st.session_state:
        st.session_state[key] = value

    # Panggil fungsi component
    component_value = _component_func(
        label=label,
        value=value,
        key=key,
        default=value,
        placeholder=placeholder,
        disabled=disabled,
    )
    return component_value
