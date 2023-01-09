
holdsAt(F,Te) :- initiatedAt(F,Ts), fluent(F), next(Ts,Te).
holdsAt(F,Te) :- holdsAt(F,Ts), not terminatedAt(F,Ts), fluent(F), next(Ts,Te).


fluent(moving(X,Y)) :- person(X), person(Y), X != Y.
fluent(meeting(X,Y)) :- person(X), person(Y), X != Y.


#include "../next.lp".

% Distance/orientation calculations:
#include "../distances.lp".

% Definitions of "type" predicates (extracted from the data):
time(T) :- happensAt(disappear(_),T).
time(T) :- happensAt(appear(_),T).
time(T) :- happensAt(active(_),T).
time(T) :- happensAt(inactive(_),T).
time(T) :- happensAt(walking(_),T).
time(T) :- happensAt(running(_),T).
time(T) :- coords(_,_,_,T).
time(T) :- orientation(_,_,T).


person(X) :- happensAt(disappear(X),_).
person(X) :- happensAt(appear(X),_).
person(X) :- happensAt(active(X),_).
person(X) :- happensAt(inactive(X),_).
person(X) :- happensAt(walking(X),_).
person(X) :- happensAt(running(X),_).
person(X) :- coords(X,_,_,_).
person(X) :- orientation(X,_,_).


% Definition for the "moving" complex event: 
initiatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),happensAt(walking(X1),X2),orientationMove(X0,X1,X2),close(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X1),X2),happensAt(inactive(X0),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X1),X2), person(X0).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).

% Definition for the "meeting" complex event:
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X0),X2),person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X1),X2),person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,25,X2).

	orientation(ms3ggt_3,148,774). coords(ms3ggt_3,60,87,774).  visible(ms3ggt_3,774). happensAt(walking(ms3ggt_3),774). orientation(ms3ggt_4,120,774). coords(ms3ggt_4,146,24,774).  visible(ms3ggt_4,774). happensAt(walking(ms3ggt_4),774).
orientation(ms3ggt_3,151,775). coords(ms3ggt_3,60,89,775).  visible(ms3ggt_3,775). happensAt(walking(ms3ggt_3),775). orientation(ms3ggt_4,120,775). coords(ms3ggt_4,146,24,775).  visible(ms3ggt_4,775). happensAt(walking(ms3ggt_4),775).
orientation(ms3ggt_3,151,776). coords(ms3ggt_3,60,89,776).  visible(ms3ggt_3,776). happensAt(walking(ms3ggt_3),776). orientation(ms3ggt_4,120,776). coords(ms3ggt_4,146,25,776).  visible(ms3ggt_4,776). happensAt(walking(ms3ggt_4),776).
orientation(ms3ggt_3,151,777). coords(ms3ggt_3,60,90,777).  visible(ms3ggt_3,777). happensAt(walking(ms3ggt_3),777). orientation(ms3ggt_4,120,777). coords(ms3ggt_4,146,25,777).  visible(ms3ggt_4,777). happensAt(walking(ms3ggt_4),777).
orientation(ms3ggt_3,151,778). coords(ms3ggt_3,60,91,778).  visible(ms3ggt_3,778). happensAt(walking(ms3ggt_3),778). orientation(ms3ggt_4,120,778). coords(ms3ggt_4,146,25,778).  visible(ms3ggt_4,778). happensAt(walking(ms3ggt_4),778).
orientation(ms3ggt_3,151,779). coords(ms3ggt_3,59,91,779).  visible(ms3ggt_3,779). happensAt(walking(ms3ggt_3),779). orientation(ms3ggt_4,120,779). coords(ms3ggt_4,146,25,779).  visible(ms3ggt_4,779). happensAt(walking(ms3ggt_4),779).
orientation(ms3ggt_3,151,780). coords(ms3ggt_3,58,92,780).  visible(ms3ggt_3,780). happensAt(walking(ms3ggt_3),780). orientation(ms3ggt_4,120,780). coords(ms3ggt_4,145,26,780).  visible(ms3ggt_4,780). happensAt(walking(ms3ggt_4),780).
orientation(ms3ggt_3,151,781). coords(ms3ggt_3,57,92,781).  visible(ms3ggt_3,781). happensAt(walking(ms3ggt_3),781). orientation(ms3ggt_4,120,781). coords(ms3ggt_4,145,26,781).  visible(ms3ggt_4,781). happensAt(walking(ms3ggt_4),781).
orientation(ms3ggt_3,151,782). coords(ms3ggt_3,57,93,782).  visible(ms3ggt_3,782). happensAt(walking(ms3ggt_3),782). orientation(ms3ggt_4,120,782). coords(ms3ggt_4,145,26,782).  visible(ms3ggt_4,782). happensAt(walking(ms3ggt_4),782).
orientation(ms3ggt_3,151,783). coords(ms3ggt_3,56,94,783).  visible(ms3ggt_3,783). happensAt(walking(ms3ggt_3),783). orientation(ms3ggt_4,120,783). coords(ms3ggt_4,145,26,783).  visible(ms3ggt_4,783). happensAt(walking(ms3ggt_4),783).
orientation(ms3ggt_3,151,784). coords(ms3ggt_3,55,94,784).  visible(ms3ggt_3,784). happensAt(walking(ms3ggt_3),784). orientation(ms3ggt_4,120,784). coords(ms3ggt_4,145,26,784).  visible(ms3ggt_4,784). happensAt(walking(ms3ggt_4),784).
orientation(ms3ggt_3,162,785). coords(ms3ggt_3,54,94,785).  visible(ms3ggt_3,785). happensAt(walking(ms3ggt_3),785). orientation(ms3ggt_4,120,785). coords(ms3ggt_4,144,26,785).  visible(ms3ggt_4,785). happensAt(walking(ms3ggt_4),785).
orientation(ms3ggt_3,162,786). coords(ms3ggt_3,54,97,786).  visible(ms3ggt_3,786). happensAt(walking(ms3ggt_3),786). orientation(ms3ggt_4,120,786). coords(ms3ggt_4,144,26,786).  visible(ms3ggt_4,786). happensAt(walking(ms3ggt_4),786).
orientation(ms3ggt_3,162,787). coords(ms3ggt_3,54,98,787).  visible(ms3ggt_3,787). happensAt(walking(ms3ggt_3),787). orientation(ms3ggt_4,120,787). coords(ms3ggt_4,143,26,787).  visible(ms3ggt_4,787). happensAt(walking(ms3ggt_4),787).
orientation(ms3ggt_3,162,788). coords(ms3ggt_3,54,98,788).  visible(ms3ggt_3,788). happensAt(walking(ms3ggt_3),788). orientation(ms3ggt_4,120,788). coords(ms3ggt_4,143,26,788).  visible(ms3ggt_4,788). happensAt(walking(ms3ggt_4),788).
orientation(ms3ggt_3,162,789). coords(ms3ggt_3,54,99,789).  visible(ms3ggt_3,789). happensAt(walking(ms3ggt_3),789). orientation(ms3ggt_4,120,789). coords(ms3ggt_4,143,26,789).  visible(ms3ggt_4,789). happensAt(walking(ms3ggt_4),789).
orientation(ms3ggt_3,162,790). coords(ms3ggt_3,53,99,790).  visible(ms3ggt_3,790). happensAt(walking(ms3ggt_3),790). orientation(ms3ggt_4,120,790). coords(ms3ggt_4,143,26,790).  visible(ms3ggt_4,790). happensAt(walking(ms3ggt_4),790).
orientation(ms3ggt_3,162,791). coords(ms3ggt_3,53,100,791).  visible(ms3ggt_3,791). happensAt(walking(ms3ggt_3),791). orientation(ms3ggt_4,120,791). coords(ms3ggt_4,143,26,791).  visible(ms3ggt_4,791). happensAt(walking(ms3ggt_4),791).
orientation(ms3ggt_3,168,792). coords(ms3ggt_3,52,100,792).  visible(ms3ggt_3,792). happensAt(walking(ms3ggt_3),792). orientation(ms3ggt_4,120,792). coords(ms3ggt_4,143,26,792).  visible(ms3ggt_4,792). happensAt(walking(ms3ggt_4),792).
orientation(ms3ggt_3,168,793). coords(ms3ggt_3,52,101,793).  visible(ms3ggt_3,793). happensAt(walking(ms3ggt_3),793). orientation(ms3ggt_4,120,793). coords(ms3ggt_4,143,26,793).  visible(ms3ggt_4,793). happensAt(walking(ms3ggt_4),793).
orientation(ms3ggt_3,150,794). coords(ms3ggt_3,52,102,794).  visible(ms3ggt_3,794). happensAt(walking(ms3ggt_3),794). orientation(ms3ggt_4,120,794). coords(ms3ggt_4,143,26,794).  visible(ms3ggt_4,794). happensAt(walking(ms3ggt_4),794).
orientation(ms3ggt_3,157,795). coords(ms3ggt_3,47,103,795).  visible(ms3ggt_3,795). happensAt(walking(ms3ggt_3),795). orientation(ms3ggt_4,120,795). coords(ms3ggt_4,142,27,795).  visible(ms3ggt_4,795). happensAt(walking(ms3ggt_4),795).
orientation(ms3ggt_3,157,796). coords(ms3ggt_3,45,105,796).  visible(ms3ggt_3,796). happensAt(walking(ms3ggt_3),796). orientation(ms3ggt_4,120,796). coords(ms3ggt_4,142,27,796).  visible(ms3ggt_4,796). happensAt(walking(ms3ggt_4),796).
orientation(ms3ggt_3,157,797). coords(ms3ggt_3,45,105,797).  visible(ms3ggt_3,797). happensAt(walking(ms3ggt_3),797). orientation(ms3ggt_4,120,797). coords(ms3ggt_4,142,27,797).  visible(ms3ggt_4,797). happensAt(walking(ms3ggt_4),797).
orientation(ms3ggt_3,157,798). coords(ms3ggt_3,45,107,798).  visible(ms3ggt_3,798). happensAt(walking(ms3ggt_3),798). orientation(ms3ggt_4,120,798). coords(ms3ggt_4,142,28,798).  visible(ms3ggt_4,798). happensAt(walking(ms3ggt_4),798).
#show holdsAt/2.