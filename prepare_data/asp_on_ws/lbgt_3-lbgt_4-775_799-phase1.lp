
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

	orientation(lbgt_3,61,775). coords(lbgt_3,146,162,775).  visible(lbgt_3,775). happensAt(inactive(lbgt_3),775). orientation(lbgt_4,146,775). coords(lbgt_4,84,114,775).  visible(lbgt_4,775). happensAt(walking(lbgt_4),775).
orientation(lbgt_3,61,776). coords(lbgt_3,146,162,776).  visible(lbgt_3,776). happensAt(inactive(lbgt_3),776). orientation(lbgt_4,146,776). coords(lbgt_4,84,114,776).  visible(lbgt_4,776). happensAt(walking(lbgt_4),776).
orientation(lbgt_3,61,777). coords(lbgt_3,146,162,777).  visible(lbgt_3,777). happensAt(inactive(lbgt_3),777). orientation(lbgt_4,146,777). coords(lbgt_4,84,114,777).  visible(lbgt_4,777). happensAt(walking(lbgt_4),777).
orientation(lbgt_3,61,778). coords(lbgt_3,146,162,778).  visible(lbgt_3,778). happensAt(inactive(lbgt_3),778). orientation(lbgt_4,146,778). coords(lbgt_4,84,114,778).  visible(lbgt_4,778). happensAt(walking(lbgt_4),778).
orientation(lbgt_3,61,779). coords(lbgt_3,146,162,779).  visible(lbgt_3,779). happensAt(inactive(lbgt_3),779). orientation(lbgt_4,146,779). coords(lbgt_4,84,113,779).  visible(lbgt_4,779). happensAt(walking(lbgt_4),779).
orientation(lbgt_3,61,780). coords(lbgt_3,146,162,780).  visible(lbgt_3,780). happensAt(inactive(lbgt_3),780). orientation(lbgt_4,146,780). coords(lbgt_4,84,113,780).  visible(lbgt_4,780). happensAt(walking(lbgt_4),780).
orientation(lbgt_3,61,781). coords(lbgt_3,146,162,781).  visible(lbgt_3,781). happensAt(inactive(lbgt_3),781). orientation(lbgt_4,146,781). coords(lbgt_4,84,113,781).  visible(lbgt_4,781). happensAt(walking(lbgt_4),781).
orientation(lbgt_3,61,782). coords(lbgt_3,146,162,782).  visible(lbgt_3,782). happensAt(inactive(lbgt_3),782). orientation(lbgt_4,146,782). coords(lbgt_4,84,112,782).  visible(lbgt_4,782). happensAt(walking(lbgt_4),782).
orientation(lbgt_3,61,783). coords(lbgt_3,146,162,783).  visible(lbgt_3,783). happensAt(inactive(lbgt_3),783). orientation(lbgt_4,146,783). coords(lbgt_4,83,111,783).  visible(lbgt_4,783). happensAt(walking(lbgt_4),783).
orientation(lbgt_3,61,784). coords(lbgt_3,146,162,784).  visible(lbgt_3,784). happensAt(inactive(lbgt_3),784). orientation(lbgt_4,149,784). coords(lbgt_4,82,109,784).  visible(lbgt_4,784). happensAt(walking(lbgt_4),784).
orientation(lbgt_3,61,785). coords(lbgt_3,146,162,785).  visible(lbgt_3,785). happensAt(inactive(lbgt_3),785). orientation(lbgt_4,151,785). coords(lbgt_4,82,107,785).  visible(lbgt_4,785). happensAt(walking(lbgt_4),785).
orientation(lbgt_3,61,786). coords(lbgt_3,146,162,786).  visible(lbgt_3,786). happensAt(inactive(lbgt_3),786). orientation(lbgt_4,151,786). coords(lbgt_4,81,107,786).  visible(lbgt_4,786). happensAt(walking(lbgt_4),786).
orientation(lbgt_3,61,787). coords(lbgt_3,146,162,787).  visible(lbgt_3,787). happensAt(inactive(lbgt_3),787). orientation(lbgt_4,151,787). coords(lbgt_4,82,107,787).  visible(lbgt_4,787). happensAt(walking(lbgt_4),787).
orientation(lbgt_3,61,788). coords(lbgt_3,146,162,788).  visible(lbgt_3,788). happensAt(inactive(lbgt_3),788). orientation(lbgt_4,151,788). coords(lbgt_4,82,106,788).  visible(lbgt_4,788). happensAt(walking(lbgt_4),788).
orientation(lbgt_3,61,789). coords(lbgt_3,146,162,789).  visible(lbgt_3,789). happensAt(inactive(lbgt_3),789). orientation(lbgt_4,145,789). coords(lbgt_4,82,106,789).  visible(lbgt_4,789). happensAt(walking(lbgt_4),789).
orientation(lbgt_3,61,790). coords(lbgt_3,146,162,790).  visible(lbgt_3,790). happensAt(inactive(lbgt_3),790). orientation(lbgt_4,145,790). coords(lbgt_4,82,105,790).  visible(lbgt_4,790). happensAt(walking(lbgt_4),790).
orientation(lbgt_3,61,791). coords(lbgt_3,146,162,791).  visible(lbgt_3,791). happensAt(inactive(lbgt_3),791). orientation(lbgt_4,141,791). coords(lbgt_4,82,105,791).  visible(lbgt_4,791). happensAt(walking(lbgt_4),791).
orientation(lbgt_3,61,792). coords(lbgt_3,146,162,792).  visible(lbgt_3,792). happensAt(inactive(lbgt_3),792). orientation(lbgt_4,139,792). coords(lbgt_4,82,105,792).  visible(lbgt_4,792). happensAt(walking(lbgt_4),792).
orientation(lbgt_3,61,793). coords(lbgt_3,146,162,793).  visible(lbgt_3,793). happensAt(inactive(lbgt_3),793). orientation(lbgt_4,139,793). coords(lbgt_4,82,104,793).  visible(lbgt_4,793). happensAt(walking(lbgt_4),793).
orientation(lbgt_3,61,794). coords(lbgt_3,146,162,794).  visible(lbgt_3,794). happensAt(inactive(lbgt_3),794). orientation(lbgt_4,139,794). coords(lbgt_4,82,103,794).  visible(lbgt_4,794). happensAt(walking(lbgt_4),794).
orientation(lbgt_3,61,795). coords(lbgt_3,146,162,795).  visible(lbgt_3,795). happensAt(inactive(lbgt_3),795). orientation(lbgt_4,139,795). coords(lbgt_4,82,101,795).  visible(lbgt_4,795). happensAt(walking(lbgt_4),795).
orientation(lbgt_3,61,796). coords(lbgt_3,146,162,796).  visible(lbgt_3,796). happensAt(inactive(lbgt_3),796). orientation(lbgt_4,139,796). coords(lbgt_4,82,100,796).  visible(lbgt_4,796). happensAt(walking(lbgt_4),796).
orientation(lbgt_3,61,797). coords(lbgt_3,146,162,797).  visible(lbgt_3,797). happensAt(inactive(lbgt_3),797). orientation(lbgt_4,139,797). coords(lbgt_4,82,98,797).  visible(lbgt_4,797). happensAt(walking(lbgt_4),797).
orientation(lbgt_3,61,798). coords(lbgt_3,146,162,798).  visible(lbgt_3,798). happensAt(inactive(lbgt_3),798). orientation(lbgt_4,139,798). coords(lbgt_4,82,97,798).  visible(lbgt_4,798). happensAt(walking(lbgt_4),798).
orientation(lbgt_3,61,799). coords(lbgt_3,146,162,799).  visible(lbgt_3,799). happensAt(inactive(lbgt_3),799). orientation(lbgt_4,139,799). coords(lbgt_4,82,96,799).  visible(lbgt_4,799). happensAt(walking(lbgt_4),799).
#show holdsAt/2.