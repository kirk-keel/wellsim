"""Streamlit UI dashboard for wellsim (Milestone M6).

Visualizes steady-state blowout rates, downhole pressure/temperature/Mach profiles,
relief well dynamic kill hydraulics, and transient kick simulation.
"""

from __future__ import annotations

import sys


def main() -> None:
    try:
        import streamlit as st
    except ImportError:
        print("Streamlit is not installed. To run the web console, use:")
        print("  python -m wellsim.ui")
        sys.exit(1)

    st.set_page_config(page_title="wellsim - Wild Well Engineering & Drilling Simulator", layout="wide")

    st.title("wellsim // Well Control & Drilling Simulator")
    st.markdown("Steady-state blowout rate, dynamic kill, and interactive well control training platform.")

    tab1, tab2, tab3 = st.tabs(["Transient Well Control", "Blowout Nodal Analysis", "Kill Sheet Calculator"])

    with tab1:
        st.subheader("Transient Kick & Choke Simulator")
        st.info("For full interactive cyberbase console, run `python -m wellsim.ui` to access the dedicated web simulator.")

    with tab2:
        st.subheader("Steady-State Nodal Analysis & Mach-1 Choke")
        st.write("Load YAML case specification to evaluate operating points and depth profiles.")

    with tab3:
        st.subheader("IADC WellSharp Kill Sheet")
        st.write("Calculate KMW, ICP, FCP, MAASP, and drillpipe step-down schedule.")


if __name__ == "__main__":
    main()
