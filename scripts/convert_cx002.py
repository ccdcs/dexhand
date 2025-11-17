from isaaclab.utils.assets import build_usd_from_urdf

URDF_PATH = "/home/msclab/msc_lab/urdf/cx002.urdf"
USD_OUTPUT = "/home/msclab/Github/dexhand/assets/cx002/cx002.usd"

def main():
    build_usd_from_urdf(
        URDF_PATH,
        USD_OUTPUT,
        make_instanceable=False,
        force=True,
    )
    print("Saved USD to:", USD_OUTPUT)

if __name__ == "__main__":
    main()


