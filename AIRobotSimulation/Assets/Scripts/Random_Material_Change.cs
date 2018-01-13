using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Random_Material_Change : MonoBehaviour
{

    public Material[] material;
    private bool isSticking = false;

    // Use this for initialization
    void Start()
    {
      Renderer rend = GetComponent<Renderer>();
      rend.enabled = true;
      int index = UnityEngine.Random.Range(0, 2);
      Debug.Log(index);
      rend.sharedMaterial = material[index];
    }

    // Update is called once per frame
    void Update()
    {

    }


}
