#pragma once


// OCT_PANEL dialog

class OCT_PANEL : public CDialogEx
{
	DECLARE_DYNAMIC(OCT_PANEL)

public:
	OCT_PANEL(CWnd* pParent = nullptr);   // standard constructor
	virtual ~OCT_PANEL();

// Dialog Data
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

	DECLARE_MESSAGE_MAP()
};
