
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

	orientation(mwt2gt_1,153,781). coords(mwt2gt_1,30,119,781).  visible(mwt2gt_1,781). happensAt(active(mwt2gt_1),781). orientation(mwt2gt_2,152,781). coords(mwt2gt_2,28,125,781).  visible(mwt2gt_2,781). happensAt(active(mwt2gt_2),781).
orientation(mwt2gt_1,153,782). coords(mwt2gt_1,30,119,782).  visible(mwt2gt_1,782). happensAt(active(mwt2gt_1),782). orientation(mwt2gt_2,152,782). coords(mwt2gt_2,28,125,782).  visible(mwt2gt_2,782). happensAt(active(mwt2gt_2),782).
orientation(mwt2gt_1,153,783). coords(mwt2gt_1,30,119,783).  visible(mwt2gt_1,783). happensAt(active(mwt2gt_1),783). orientation(mwt2gt_2,152,783). coords(mwt2gt_2,28,125,783).  visible(mwt2gt_2,783). happensAt(active(mwt2gt_2),783).
orientation(mwt2gt_1,153,784). coords(mwt2gt_1,30,119,784).  visible(mwt2gt_1,784). happensAt(active(mwt2gt_1),784). orientation(mwt2gt_2,152,784). coords(mwt2gt_2,28,125,784).  visible(mwt2gt_2,784). happensAt(active(mwt2gt_2),784).
orientation(mwt2gt_1,153,785). coords(mwt2gt_1,30,119,785).  visible(mwt2gt_1,785). happensAt(active(mwt2gt_1),785). orientation(mwt2gt_2,152,785). coords(mwt2gt_2,28,125,785).  visible(mwt2gt_2,785). happensAt(active(mwt2gt_2),785).
orientation(mwt2gt_1,153,786). coords(mwt2gt_1,30,119,786).  visible(mwt2gt_1,786). happensAt(active(mwt2gt_1),786). orientation(mwt2gt_2,152,786). coords(mwt2gt_2,28,125,786).  visible(mwt2gt_2,786). happensAt(active(mwt2gt_2),786).
orientation(mwt2gt_1,153,787). coords(mwt2gt_1,30,121,787).  visible(mwt2gt_1,787). happensAt(active(mwt2gt_1),787). orientation(mwt2gt_2,152,787). coords(mwt2gt_2,28,125,787).  visible(mwt2gt_2,787). happensAt(active(mwt2gt_2),787).
orientation(mwt2gt_1,153,788). coords(mwt2gt_1,30,121,788).  visible(mwt2gt_1,788). happensAt(active(mwt2gt_1),788). orientation(mwt2gt_2,152,788). coords(mwt2gt_2,28,126,788).  visible(mwt2gt_2,788). happensAt(active(mwt2gt_2),788).
orientation(mwt2gt_1,153,789). coords(mwt2gt_1,30,121,789).  visible(mwt2gt_1,789). happensAt(active(mwt2gt_1),789). orientation(mwt2gt_2,152,789). coords(mwt2gt_2,28,126,789).  visible(mwt2gt_2,789). happensAt(active(mwt2gt_2),789).
orientation(mwt2gt_1,153,790). coords(mwt2gt_1,30,121,790).  visible(mwt2gt_1,790). happensAt(active(mwt2gt_1),790). orientation(mwt2gt_2,152,790). coords(mwt2gt_2,28,126,790).  visible(mwt2gt_2,790). happensAt(active(mwt2gt_2),790).
orientation(mwt2gt_1,153,791). coords(mwt2gt_1,30,121,791).  visible(mwt2gt_1,791). happensAt(active(mwt2gt_1),791). orientation(mwt2gt_2,152,791). coords(mwt2gt_2,28,126,791).  visible(mwt2gt_2,791). happensAt(active(mwt2gt_2),791).
orientation(mwt2gt_1,153,792). coords(mwt2gt_1,30,122,792).  visible(mwt2gt_1,792). happensAt(active(mwt2gt_1),792). orientation(mwt2gt_2,152,792). coords(mwt2gt_2,28,127,792).  visible(mwt2gt_2,792). happensAt(active(mwt2gt_2),792).
orientation(mwt2gt_1,153,793). coords(mwt2gt_1,30,122,793).  visible(mwt2gt_1,793). happensAt(active(mwt2gt_1),793). orientation(mwt2gt_2,152,793). coords(mwt2gt_2,28,125,793).  visible(mwt2gt_2,793). happensAt(active(mwt2gt_2),793).
orientation(mwt2gt_1,153,794). coords(mwt2gt_1,30,122,794).  visible(mwt2gt_1,794). happensAt(active(mwt2gt_1),794). orientation(mwt2gt_2,152,794). coords(mwt2gt_2,28,125,794).  visible(mwt2gt_2,794). happensAt(active(mwt2gt_2),794).
orientation(mwt2gt_1,153,795). coords(mwt2gt_1,30,122,795).  visible(mwt2gt_1,795). happensAt(active(mwt2gt_1),795). orientation(mwt2gt_2,152,795). coords(mwt2gt_2,28,125,795).  visible(mwt2gt_2,795). happensAt(active(mwt2gt_2),795).
orientation(mwt2gt_1,153,796). coords(mwt2gt_1,30,122,796).  visible(mwt2gt_1,796). happensAt(active(mwt2gt_1),796). orientation(mwt2gt_2,152,796). coords(mwt2gt_2,28,125,796).  visible(mwt2gt_2,796). happensAt(active(mwt2gt_2),796).
orientation(mwt2gt_1,153,797). coords(mwt2gt_1,30,122,797).  visible(mwt2gt_1,797). happensAt(active(mwt2gt_1),797). orientation(mwt2gt_2,152,797). coords(mwt2gt_2,28,125,797).  visible(mwt2gt_2,797). happensAt(active(mwt2gt_2),797).
orientation(mwt2gt_1,153,798). coords(mwt2gt_1,30,122,798).  visible(mwt2gt_1,798). happensAt(active(mwt2gt_1),798). orientation(mwt2gt_2,152,798). coords(mwt2gt_2,28,125,798).  visible(mwt2gt_2,798). happensAt(active(mwt2gt_2),798).
orientation(mwt2gt_1,153,799). coords(mwt2gt_1,30,122,799).  visible(mwt2gt_1,799). happensAt(active(mwt2gt_1),799). orientation(mwt2gt_2,152,799). coords(mwt2gt_2,28,125,799).  visible(mwt2gt_2,799). happensAt(active(mwt2gt_2),799).
orientation(mwt2gt_1,153,800). coords(mwt2gt_1,30,122,800).  visible(mwt2gt_1,800). happensAt(active(mwt2gt_1),800). orientation(mwt2gt_2,152,800). coords(mwt2gt_2,28,125,800).  visible(mwt2gt_2,800). happensAt(active(mwt2gt_2),800).
orientation(mwt2gt_1,153,801). coords(mwt2gt_1,30,122,801).  visible(mwt2gt_1,801). happensAt(active(mwt2gt_1),801). orientation(mwt2gt_2,152,801). coords(mwt2gt_2,28,125,801).  visible(mwt2gt_2,801). happensAt(active(mwt2gt_2),801).
orientation(mwt2gt_1,153,802). coords(mwt2gt_1,30,122,802).  visible(mwt2gt_1,802). happensAt(active(mwt2gt_1),802). orientation(mwt2gt_2,152,802). coords(mwt2gt_2,28,125,802).  visible(mwt2gt_2,802). happensAt(active(mwt2gt_2),802).
orientation(mwt2gt_1,153,803). coords(mwt2gt_1,30,122,803).  visible(mwt2gt_1,803). happensAt(active(mwt2gt_1),803). orientation(mwt2gt_2,152,803). coords(mwt2gt_2,28,125,803).  visible(mwt2gt_2,803). happensAt(active(mwt2gt_2),803).
orientation(mwt2gt_1,153,804). coords(mwt2gt_1,30,122,804).  visible(mwt2gt_1,804). happensAt(active(mwt2gt_1),804). orientation(mwt2gt_2,152,804). coords(mwt2gt_2,28,125,804).  visible(mwt2gt_2,804). happensAt(active(mwt2gt_2),804).
orientation(mwt2gt_1,153,805). coords(mwt2gt_1,30,122,805).  visible(mwt2gt_1,805). happensAt(active(mwt2gt_1),805). orientation(mwt2gt_2,152,805). coords(mwt2gt_2,28,125,805).  visible(mwt2gt_2,805). happensAt(active(mwt2gt_2),805).
#show holdsAt/2.