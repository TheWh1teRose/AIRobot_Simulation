using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using System.Threading;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System;

public class UnityPythonConector : MonoBehaviour {

    public GameObject traget;
    private UdpClient client = null;
    private UdpClient rcvClient = null;
    private Thread receiveThread;
    private float[,,] positionMatrix;
    private int isRestarted = 1;
    private float timer = 0;
    private float sendTimer = 0;
    private bool isRunThread = true;

    public float moveSpeed = 0.35f;

     int horizontal = 0;
     int vertical = 0;
     int height = 0;

    private float x = 60;
    private float y = 60;
    private float h = 60;
    private int smothing = 8;
    //                         x,   y,    h
    private float[] startPositions = {-1.3f,0.5f,1.3f};
    private float[] endPositions   = {1.1f,2.9f,3.7f};

    // Use this for initialization
    void Start () {
        client = new UdpClient(5002);
        rcvClient = new UdpClient(5003);

        //positionMatrix = new float[Convert.ToInt32(x), Convert.ToInt32(y), Convert.ToInt32(h)];
        //receiveThread = new Thread(new ThreadStart(RemoteControl));
        //receiveThread.Start();
    }

	// Update is called once per frame
	void Update () {
        if(Input.GetKeyDown("f") && timer > 1){
            SceneManager.LoadScene(SceneManager.GetActiveScene().name);
            client = new UdpClient(5002);
            //rcvClient = new UdpClient(5003);
            Time.timeScale = 1f;
            isRestarted = 1;
            timer = 0;
        }

        timer += Time.deltaTime;
        sendTimer += Time.deltaTime;

        Debug.Log("" + horizontal + ":" + vertical + ":" + height);

        traget.transform.Translate(moveSpeed * horizontal*Time.deltaTime,
            moveSpeed * vertical * Time.deltaTime,
            moveSpeed * height * Time.deltaTime);

        UDPSendControls();
    }

    void OnApplicationQuit()
    {
        client.Close();
        rcvClient.Close();
        isRunThread = false;
    }

    private void UDPSendControls()
    {
        if(sendTimer>0.15)
        {

          int horizontal = 0;
          int vertical = 0;
          int height = 0;
          if(Input.GetAxis("Horizontal")>0){horizontal=1;}
          else if(Input.GetAxis("Horizontal")<0){horizontal=-1;}
          else if(Input.GetAxis("Vertical")>0){vertical=1;}
          else if(Input.GetAxis("Vertical")<0){vertical=-1;}
          else if(Input.GetAxis("Height")>0){height=1;}
          else if(Input.GetAxis("Height")<0){height=-1;}

            Debug.Log(isRestarted);

            string controls = horizontal + ":"
            + -vertical + ":"
            + -height + ":"  + getGrab();

            Debug.Log(controls);

            IPEndPoint endpoint = new IPEndPoint(IPAddress.Parse("127.0.0.1"), 5002);
            byte[] sendBytes = Encoding.ASCII.GetBytes(controls + "$" + isRestarted + "$" + getPositionInMatrix());
            isRestarted = 0;
            Debug.Log(getPositionInMatrix());
            client.Send(sendBytes, sendBytes.Length, endpoint);
            sendTimer=0;
        }
    }

    private void RemoteControl()
    {
          while(isRunThread)
          {
            IPEndPoint endpoint = new IPEndPoint(IPAddress.Parse("127.0.0.1"), 5003);
            Byte[] receiveBytes = rcvClient.Receive(ref endpoint);
            string returnData = Encoding.ASCII.GetString(receiveBytes);
            if(returnData[0].Equals('1')){horizontal=1;}
            if(returnData[2].Equals('1')){horizontal=-1;}
            if(returnData[4].Equals('1')){vertical=1;}
            if(returnData[6].Equals('1')){vertical=-1;}
            if(returnData[8].Equals('1')){height=1;}
            if(returnData[10].Equals('1')){height=-1;}

            if(returnData[0].Equals('0')&&returnData[2].Equals('0')){horizontal=0;}
            if(returnData[4].Equals('0')&&returnData[6].Equals('0')){vertical=0;}
            if(returnData[8].Equals('0')&&returnData[10].Equals('0')){height=0;}
            //if(returnData[6].Equals('1')){horizontal=1;}
            //Sif(returnData[7].Equals('1')){horizontal=1;}
            Debug.Log(returnData);
          }

    }

    private String getPositionInMatrix()
    {
        //get the postions from the target. NOTE: some dimentions are switcht for realism
        float targetX = traget.transform.position.x;
        float targetY = traget.transform.position.z;
        float targetH = traget.transform.position.y;

        float distanceToStartX = targetX - startPositions[0];
        float distanceToStartY = targetY - startPositions[1];
        float distanceToStartH = targetH - startPositions[2];

        int positionInPosiotinMatrixX = Convert.ToInt32(Math.Round(distanceToStartX / ((endPositions[0] - startPositions[0]) / x)));
        int positionInPosiotinMatrixY = Convert.ToInt32(Math.Round(distanceToStartY / ((endPositions[1] - startPositions[1]) / y)));
        int positionInPosiotinMatrixH = Convert.ToInt32(Math.Round(distanceToStartH / ((endPositions[2] - startPositions[2]) / h)));

        //Debug.Log(positionInPosiotinMatrixX + ":" + positionInPosiotinMatrixY + ":" + positionInPosiotinMatrixH);

        return (positionInPosiotinMatrixX + ":" + positionInPosiotinMatrixY + ":" + positionInPosiotinMatrixH);

    }

    private int getGrab()
    {
        if (Input.GetKey("g"))
        {
            return 1;
        }else if (Input.GetKey("b"))
        {
            return -1;
        }
        return 0;
    }
}
