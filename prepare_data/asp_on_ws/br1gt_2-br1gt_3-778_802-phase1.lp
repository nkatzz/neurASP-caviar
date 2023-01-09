
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

	orientation(br1gt_2,12,778). coords(br1gt_2,86,209,778).  visible(br1gt_2,778). happensAt(inactive(br1gt_2),778). orientation(br1gt_3,107,778). coords(br1gt_3,302,136,778).  visible(br1gt_3,778). happensAt(walking(br1gt_3),778).
orientation(br1gt_2,12,779). coords(br1gt_2,86,209,779).  visible(br1gt_2,779). happensAt(inactive(br1gt_2),779). orientation(br1gt_3,107,779). coords(br1gt_3,302,137,779).  visible(br1gt_3,779). happensAt(walking(br1gt_3),779).
orientation(br1gt_2,12,780). coords(br1gt_2,86,209,780).  visible(br1gt_2,780). happensAt(inactive(br1gt_2),780). orientation(br1gt_3,107,780). coords(br1gt_3,301,138,780).  visible(br1gt_3,780). happensAt(walking(br1gt_3),780).
orientation(br1gt_2,12,781). coords(br1gt_2,86,209,781).  visible(br1gt_2,781). happensAt(inactive(br1gt_2),781). orientation(br1gt_3,107,781). coords(br1gt_3,301,139,781).  visible(br1gt_3,781). happensAt(walking(br1gt_3),781).
orientation(br1gt_2,12,782). coords(br1gt_2,86,209,782).  visible(br1gt_2,782). happensAt(inactive(br1gt_2),782). orientation(br1gt_3,107,782). coords(br1gt_3,301,143,782).  visible(br1gt_3,782). happensAt(walking(br1gt_3),782).
orientation(br1gt_2,12,783). coords(br1gt_2,86,209,783).  visible(br1gt_2,783). happensAt(inactive(br1gt_2),783). orientation(br1gt_3,106,783). coords(br1gt_3,301,148,783).  visible(br1gt_3,783). happensAt(walking(br1gt_3),783).
orientation(br1gt_2,12,784). coords(br1gt_2,86,209,784).  visible(br1gt_2,784). happensAt(inactive(br1gt_2),784). orientation(br1gt_3,102,784). coords(br1gt_3,301,151,784).  visible(br1gt_3,784). happensAt(walking(br1gt_3),784).
orientation(br1gt_2,12,785). coords(br1gt_2,86,209,785).  visible(br1gt_2,785). happensAt(inactive(br1gt_2),785). orientation(br1gt_3,102,785). coords(br1gt_3,301,155,785).  visible(br1gt_3,785). happensAt(walking(br1gt_3),785).
orientation(br1gt_2,12,786). coords(br1gt_2,86,209,786).  visible(br1gt_2,786). happensAt(inactive(br1gt_2),786). orientation(br1gt_3,102,786). coords(br1gt_3,301,157,786).  visible(br1gt_3,786). happensAt(walking(br1gt_3),786).
orientation(br1gt_2,12,787). coords(br1gt_2,86,209,787).  visible(br1gt_2,787). happensAt(inactive(br1gt_2),787). orientation(br1gt_3,99,787). coords(br1gt_3,301,159,787).  visible(br1gt_3,787). happensAt(walking(br1gt_3),787).
orientation(br1gt_2,12,788). coords(br1gt_2,86,209,788).  visible(br1gt_2,788). happensAt(inactive(br1gt_2),788). orientation(br1gt_3,99,788). coords(br1gt_3,300,161,788).  visible(br1gt_3,788). happensAt(walking(br1gt_3),788).
orientation(br1gt_2,12,789). coords(br1gt_2,86,209,789).  visible(br1gt_2,789). happensAt(inactive(br1gt_2),789). orientation(br1gt_3,99,789). coords(br1gt_3,300,163,789).  visible(br1gt_3,789). happensAt(walking(br1gt_3),789).
orientation(br1gt_2,12,790). coords(br1gt_2,86,209,790).  visible(br1gt_2,790). happensAt(inactive(br1gt_2),790). orientation(br1gt_3,99,790). coords(br1gt_3,299,163,790).  visible(br1gt_3,790). happensAt(walking(br1gt_3),790).
orientation(br1gt_2,12,791). coords(br1gt_2,86,209,791).  visible(br1gt_2,791). happensAt(inactive(br1gt_2),791). orientation(br1gt_3,99,791). coords(br1gt_3,298,165,791).  visible(br1gt_3,791). happensAt(walking(br1gt_3),791).
orientation(br1gt_2,12,792). coords(br1gt_2,86,209,792).  visible(br1gt_2,792). happensAt(inactive(br1gt_2),792). orientation(br1gt_3,99,792). coords(br1gt_3,298,166,792).  visible(br1gt_3,792). happensAt(walking(br1gt_3),792).
orientation(br1gt_2,12,793). coords(br1gt_2,86,209,793).  visible(br1gt_2,793). happensAt(inactive(br1gt_2),793). orientation(br1gt_3,97,793). coords(br1gt_3,297,167,793).  visible(br1gt_3,793). happensAt(walking(br1gt_3),793).
orientation(br1gt_2,12,794). coords(br1gt_2,86,209,794).  visible(br1gt_2,794). happensAt(inactive(br1gt_2),794). orientation(br1gt_3,97,794). coords(br1gt_3,297,168,794).  visible(br1gt_3,794). happensAt(walking(br1gt_3),794).
orientation(br1gt_2,12,795). coords(br1gt_2,86,209,795).  visible(br1gt_2,795). happensAt(inactive(br1gt_2),795). orientation(br1gt_3,97,795). coords(br1gt_3,296,169,795).  visible(br1gt_3,795). happensAt(walking(br1gt_3),795).
orientation(br1gt_2,12,796). coords(br1gt_2,86,209,796).  visible(br1gt_2,796). happensAt(inactive(br1gt_2),796). orientation(br1gt_3,97,796). coords(br1gt_3,296,170,796).  visible(br1gt_3,796). happensAt(walking(br1gt_3),796).
orientation(br1gt_2,12,797). coords(br1gt_2,86,209,797).  visible(br1gt_2,797). happensAt(inactive(br1gt_2),797). orientation(br1gt_3,112,797). coords(br1gt_3,296,175,797).  visible(br1gt_3,797). happensAt(walking(br1gt_3),797).
orientation(br1gt_2,12,798). coords(br1gt_2,86,209,798).  visible(br1gt_2,798). happensAt(inactive(br1gt_2),798). orientation(br1gt_3,112,798). coords(br1gt_3,296,179,798).  visible(br1gt_3,798). happensAt(walking(br1gt_3),798).
orientation(br1gt_2,12,799). coords(br1gt_2,86,209,799).  visible(br1gt_2,799). happensAt(inactive(br1gt_2),799). orientation(br1gt_3,112,799). coords(br1gt_3,294,183,799).  visible(br1gt_3,799). happensAt(walking(br1gt_3),799).
orientation(br1gt_2,12,800). coords(br1gt_2,86,209,800).  visible(br1gt_2,800). happensAt(inactive(br1gt_2),800). orientation(br1gt_3,112,800). coords(br1gt_3,294,185,800).  visible(br1gt_3,800). happensAt(walking(br1gt_3),800).
orientation(br1gt_2,12,801). coords(br1gt_2,86,209,801).  visible(br1gt_2,801). happensAt(inactive(br1gt_2),801). orientation(br1gt_3,108,801). coords(br1gt_3,294,188,801).  visible(br1gt_3,801). happensAt(walking(br1gt_3),801).
orientation(br1gt_2,12,802). coords(br1gt_2,86,209,802).  visible(br1gt_2,802). happensAt(inactive(br1gt_2),802). orientation(br1gt_3,108,802). coords(br1gt_3,294,190,802).  visible(br1gt_3,802). happensAt(walking(br1gt_3),802).
#show holdsAt/2.