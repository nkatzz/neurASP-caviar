
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

	orientation(lb2gt_5,132,783). coords(lb2gt_5,151,40,783).  visible(lb2gt_5,783). happensAt(walking(lb2gt_5),783). orientation(lb2gt_6,151,783). coords(lb2gt_6,72,122,783).  visible(lb2gt_6,783). happensAt(inactive(lb2gt_6),783).
orientation(lb2gt_5,136,784). coords(lb2gt_5,151,40,784).  visible(lb2gt_5,784). happensAt(walking(lb2gt_5),784). orientation(lb2gt_6,151,784). coords(lb2gt_6,72,122,784).  visible(lb2gt_6,784). happensAt(inactive(lb2gt_6),784).
orientation(lb2gt_5,144,785). coords(lb2gt_5,151,41,785).  visible(lb2gt_5,785). happensAt(walking(lb2gt_5),785). orientation(lb2gt_6,151,785). coords(lb2gt_6,72,122,785).  visible(lb2gt_6,785). happensAt(inactive(lb2gt_6),785).
orientation(lb2gt_5,147,786). coords(lb2gt_5,150,41,786).  visible(lb2gt_5,786). happensAt(walking(lb2gt_5),786). orientation(lb2gt_6,151,786). coords(lb2gt_6,72,122,786).  visible(lb2gt_6,786). happensAt(inactive(lb2gt_6),786).
orientation(lb2gt_5,151,787). coords(lb2gt_5,149,41,787).  visible(lb2gt_5,787). happensAt(walking(lb2gt_5),787). orientation(lb2gt_6,151,787). coords(lb2gt_6,72,122,787).  visible(lb2gt_6,787). happensAt(inactive(lb2gt_6),787).
orientation(lb2gt_5,156,788). coords(lb2gt_5,149,42,788).  visible(lb2gt_5,788). happensAt(walking(lb2gt_5),788). orientation(lb2gt_6,151,788). coords(lb2gt_6,72,122,788).  visible(lb2gt_6,788). happensAt(inactive(lb2gt_6),788).
orientation(lb2gt_5,157,789). coords(lb2gt_5,147,42,789).  visible(lb2gt_5,789). happensAt(walking(lb2gt_5),789). orientation(lb2gt_6,151,789). coords(lb2gt_6,72,122,789).  visible(lb2gt_6,789). happensAt(inactive(lb2gt_6),789).
orientation(lb2gt_5,156,790). coords(lb2gt_5,145,42,790).  visible(lb2gt_5,790). happensAt(walking(lb2gt_5),790). orientation(lb2gt_6,151,790). coords(lb2gt_6,72,122,790).  visible(lb2gt_6,790). happensAt(inactive(lb2gt_6),790).
orientation(lb2gt_5,159,791). coords(lb2gt_5,145,42,791).  visible(lb2gt_5,791). happensAt(walking(lb2gt_5),791). orientation(lb2gt_6,151,791). coords(lb2gt_6,72,122,791).  visible(lb2gt_6,791). happensAt(inactive(lb2gt_6),791).
orientation(lb2gt_5,155,792). coords(lb2gt_5,144,43,792).  visible(lb2gt_5,792). happensAt(walking(lb2gt_5),792). orientation(lb2gt_6,151,792). coords(lb2gt_6,72,122,792).  visible(lb2gt_6,792). happensAt(inactive(lb2gt_6),792).
orientation(lb2gt_5,162,793). coords(lb2gt_5,143,43,793).  visible(lb2gt_5,793). happensAt(walking(lb2gt_5),793). orientation(lb2gt_6,151,793). coords(lb2gt_6,72,122,793).  visible(lb2gt_6,793). happensAt(inactive(lb2gt_6),793).
orientation(lb2gt_5,162,794). coords(lb2gt_5,142,44,794).  visible(lb2gt_5,794). happensAt(walking(lb2gt_5),794). orientation(lb2gt_6,151,794). coords(lb2gt_6,72,122,794).  visible(lb2gt_6,794). happensAt(inactive(lb2gt_6),794).
orientation(lb2gt_5,161,795). coords(lb2gt_5,142,44,795).  visible(lb2gt_5,795). happensAt(walking(lb2gt_5),795). orientation(lb2gt_6,151,795). coords(lb2gt_6,72,122,795).  visible(lb2gt_6,795). happensAt(inactive(lb2gt_6),795).
orientation(lb2gt_5,163,796). coords(lb2gt_5,141,46,796).  visible(lb2gt_5,796). happensAt(walking(lb2gt_5),796). orientation(lb2gt_6,151,796). coords(lb2gt_6,72,122,796).  visible(lb2gt_6,796). happensAt(inactive(lb2gt_6),796).
orientation(lb2gt_5,155,797). coords(lb2gt_5,139,46,797).  visible(lb2gt_5,797). happensAt(walking(lb2gt_5),797). orientation(lb2gt_6,151,797). coords(lb2gt_6,72,122,797).  visible(lb2gt_6,797). happensAt(inactive(lb2gt_6),797).
orientation(lb2gt_5,148,798). coords(lb2gt_5,139,47,798).  visible(lb2gt_5,798). happensAt(walking(lb2gt_5),798). orientation(lb2gt_6,151,798). coords(lb2gt_6,72,122,798).  visible(lb2gt_6,798). happensAt(inactive(lb2gt_6),798).
orientation(lb2gt_5,152,799). coords(lb2gt_5,137,47,799).  visible(lb2gt_5,799). happensAt(walking(lb2gt_5),799). orientation(lb2gt_6,151,799). coords(lb2gt_6,72,122,799).  visible(lb2gt_6,799). happensAt(inactive(lb2gt_6),799).
orientation(lb2gt_5,152,800). coords(lb2gt_5,135,47,800).  visible(lb2gt_5,800). happensAt(walking(lb2gt_5),800). orientation(lb2gt_6,151,800). coords(lb2gt_6,72,122,800).  visible(lb2gt_6,800). happensAt(inactive(lb2gt_6),800).
orientation(lb2gt_5,152,801). coords(lb2gt_5,133,48,801).  visible(lb2gt_5,801). happensAt(walking(lb2gt_5),801). orientation(lb2gt_6,151,801). coords(lb2gt_6,72,122,801).  visible(lb2gt_6,801). happensAt(inactive(lb2gt_6),801).
orientation(lb2gt_5,152,802). coords(lb2gt_5,131,48,802).  visible(lb2gt_5,802). happensAt(walking(lb2gt_5),802). orientation(lb2gt_6,151,802). coords(lb2gt_6,72,122,802).  visible(lb2gt_6,802). happensAt(inactive(lb2gt_6),802).
orientation(lb2gt_5,152,803). coords(lb2gt_5,131,49,803).  visible(lb2gt_5,803). happensAt(walking(lb2gt_5),803). orientation(lb2gt_6,151,803). coords(lb2gt_6,72,122,803).  visible(lb2gt_6,803). happensAt(inactive(lb2gt_6),803).
orientation(lb2gt_5,152,804). coords(lb2gt_5,130,49,804).  visible(lb2gt_5,804). happensAt(walking(lb2gt_5),804). orientation(lb2gt_6,151,804). coords(lb2gt_6,72,122,804).  visible(lb2gt_6,804). happensAt(inactive(lb2gt_6),804).
orientation(lb2gt_5,152,805). coords(lb2gt_5,129,50,805).  visible(lb2gt_5,805). happensAt(walking(lb2gt_5),805). orientation(lb2gt_6,151,805). coords(lb2gt_6,72,122,805).  visible(lb2gt_6,805). happensAt(inactive(lb2gt_6),805).
orientation(lb2gt_5,152,806). coords(lb2gt_5,129,51,806).  visible(lb2gt_5,806). happensAt(walking(lb2gt_5),806). orientation(lb2gt_6,151,806). coords(lb2gt_6,72,122,806).  visible(lb2gt_6,806). happensAt(inactive(lb2gt_6),806).
orientation(lb2gt_5,152,807). coords(lb2gt_5,128,52,807).  visible(lb2gt_5,807). happensAt(walking(lb2gt_5),807). orientation(lb2gt_6,151,807). coords(lb2gt_6,72,122,807).  visible(lb2gt_6,807). happensAt(inactive(lb2gt_6),807).
#show holdsAt/2.