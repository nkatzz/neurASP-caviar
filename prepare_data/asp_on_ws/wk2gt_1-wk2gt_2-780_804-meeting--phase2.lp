

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
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,34,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,34,X2).


happensAt(active(X),T) :- simpleEvent(T,X,active).
happensAt(inactive(X),T) :- simpleEvent(T,X,inactive).
happensAt(walking(X),T) :- simpleEvent(T,X,walking).
happensAt(running(X),T) :- simpleEvent(T,X,running). 


holdsByInertia(moving(X,Y),T+1) :- holdsAt(moving(X,Y),T+1), not initiatedAt(moving(X,Y),T).
holdsByInertia(meeting(X,Y),T+1) :- holdsAt(meeting(X,Y),T+1), not initiatedAt(meeting(X,Y),T).

:- holdsByInertia(moving(X,Y),T+1), simpleEvent(T,X,E1), simpleEvent(T+1,X,E2), E1 != E2. 
:- holdsByInertia(meeting(X,Y),T+1), simpleEvent(T,X,E1), simpleEvent(T+1,X,E2), E1 != E2.

	person(wk2gt_1;wk2gt_2).
orientation(wk2gt_1,125,780). coords(wk2gt_1,115,53,780).  visible(wk2gt_1,780). orientation(wk2gt_2,142,780). coords(wk2gt_2,99,63,780).  visible(wk2gt_2,780).
orientation(wk2gt_1,125,781). coords(wk2gt_1,115,53,781).  visible(wk2gt_1,781). orientation(wk2gt_2,142,781). coords(wk2gt_2,99,63,781).  visible(wk2gt_2,781).
orientation(wk2gt_1,125,782). coords(wk2gt_1,115,53,782).  visible(wk2gt_1,782). orientation(wk2gt_2,142,782). coords(wk2gt_2,99,63,782).  visible(wk2gt_2,782).
orientation(wk2gt_1,125,783). coords(wk2gt_1,115,53,783).  visible(wk2gt_1,783). orientation(wk2gt_2,142,783). coords(wk2gt_2,99,63,783).  visible(wk2gt_2,783).
orientation(wk2gt_1,125,784). coords(wk2gt_1,115,53,784).  visible(wk2gt_1,784). orientation(wk2gt_2,142,784). coords(wk2gt_2,99,63,784).  visible(wk2gt_2,784).
orientation(wk2gt_1,125,785). coords(wk2gt_1,115,53,785).  visible(wk2gt_1,785). orientation(wk2gt_2,142,785). coords(wk2gt_2,99,63,785).  visible(wk2gt_2,785).
orientation(wk2gt_1,125,786). coords(wk2gt_1,115,53,786).  visible(wk2gt_1,786). orientation(wk2gt_2,142,786). coords(wk2gt_2,99,63,786).  visible(wk2gt_2,786).
orientation(wk2gt_1,125,787). coords(wk2gt_1,115,53,787).  visible(wk2gt_1,787). orientation(wk2gt_2,142,787). coords(wk2gt_2,99,63,787).  visible(wk2gt_2,787).
orientation(wk2gt_1,125,788). coords(wk2gt_1,115,53,788).  visible(wk2gt_1,788). orientation(wk2gt_2,142,788). coords(wk2gt_2,99,63,788).  visible(wk2gt_2,788).
orientation(wk2gt_1,125,789). coords(wk2gt_1,115,53,789).  visible(wk2gt_1,789). orientation(wk2gt_2,142,789). coords(wk2gt_2,99,63,789).  visible(wk2gt_2,789).
orientation(wk2gt_1,125,790). coords(wk2gt_1,115,53,790).  visible(wk2gt_1,790). orientation(wk2gt_2,142,790). coords(wk2gt_2,99,63,790).  visible(wk2gt_2,790).
orientation(wk2gt_1,125,791). coords(wk2gt_1,115,53,791).  visible(wk2gt_1,791). orientation(wk2gt_2,142,791). coords(wk2gt_2,99,63,791).  visible(wk2gt_2,791).
orientation(wk2gt_1,125,792). coords(wk2gt_1,115,53,792).  visible(wk2gt_1,792). orientation(wk2gt_2,142,792). coords(wk2gt_2,99,63,792).  visible(wk2gt_2,792).
orientation(wk2gt_1,125,793). coords(wk2gt_1,115,53,793).  visible(wk2gt_1,793). orientation(wk2gt_2,142,793). coords(wk2gt_2,99,63,793).  visible(wk2gt_2,793).
orientation(wk2gt_1,125,794). coords(wk2gt_1,115,53,794).  visible(wk2gt_1,794). orientation(wk2gt_2,142,794). coords(wk2gt_2,99,63,794).  visible(wk2gt_2,794).
orientation(wk2gt_1,125,795). coords(wk2gt_1,115,53,795).  visible(wk2gt_1,795). orientation(wk2gt_2,142,795). coords(wk2gt_2,99,63,795).  visible(wk2gt_2,795).
orientation(wk2gt_1,125,796). coords(wk2gt_1,115,53,796).  visible(wk2gt_1,796). orientation(wk2gt_2,142,796). coords(wk2gt_2,99,63,796).  visible(wk2gt_2,796).
orientation(wk2gt_1,125,797). coords(wk2gt_1,115,53,797).  visible(wk2gt_1,797). orientation(wk2gt_2,142,797). coords(wk2gt_2,99,63,797).  visible(wk2gt_2,797).
orientation(wk2gt_1,125,798). coords(wk2gt_1,115,53,798).  visible(wk2gt_1,798). orientation(wk2gt_2,142,798). coords(wk2gt_2,99,63,798).  visible(wk2gt_2,798).
orientation(wk2gt_1,125,799). coords(wk2gt_1,115,53,799).  visible(wk2gt_1,799). orientation(wk2gt_2,142,799). coords(wk2gt_2,99,63,799).  visible(wk2gt_2,799).
orientation(wk2gt_1,125,800). coords(wk2gt_1,115,53,800).  visible(wk2gt_1,800). orientation(wk2gt_2,142,800). coords(wk2gt_2,99,63,800).  visible(wk2gt_2,800).
orientation(wk2gt_1,125,801). coords(wk2gt_1,115,53,801).  visible(wk2gt_1,801). orientation(wk2gt_2,142,801). coords(wk2gt_2,99,63,801).  visible(wk2gt_2,801).
orientation(wk2gt_1,125,802). coords(wk2gt_1,115,53,802).  visible(wk2gt_1,802). orientation(wk2gt_2,142,802). coords(wk2gt_2,99,63,802).  visible(wk2gt_2,802).
orientation(wk2gt_1,125,803). coords(wk2gt_1,115,53,803).  visible(wk2gt_1,803). orientation(wk2gt_2,142,803). coords(wk2gt_2,99,63,803).  visible(wk2gt_2,803).
orientation(wk2gt_1,125,804). coords(wk2gt_1,115,53,804).  visible(wk2gt_1,804). orientation(wk2gt_2,142,804). coords(wk2gt_2,99,63,804).  visible(wk2gt_2,804).
orientation(wk2gt_1,125,805). coords(wk2gt_1,115,53,805).  visible(wk2gt_1,805). orientation(wk2gt_2,142,805). coords(wk2gt_2,99,63,805).  visible(wk2gt_2,805).
happensAt(disappear(wk2gt_1),804). happensAt(disappear(wk2gt_2),804).
1{simpleEvent(T,wk2gt_1,active); simpleEvent(T,wk2gt_1,inactive); simpleEvent(T,wk2gt_1,walking); simpleEvent(T,wk2gt_1,running)}1 :- T=780..804.
1{simpleEvent(T,wk2gt_2,active); simpleEvent(T,wk2gt_2,inactive); simpleEvent(T,wk2gt_2,walking); simpleEvent(T,wk2gt_2,running)}1 :- T=780..804.
:- holdsAt(meeting(wk2gt_1,wk2gt_2),780). :- holdsAt(meeting(wk2gt_2,wk2gt_1),780).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),781). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),781).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),782). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),782).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),783). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),783).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),784). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),784).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),785). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),785).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),786). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),786).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),787). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),787).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),788). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),788).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),789). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),789).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),790). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),790).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),791). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),791).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),792). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),792).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),793). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),793).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),794). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),794).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),795). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),795).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),796). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),796).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),797). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),797).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),798). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),798).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),799). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),799).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),800). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),800).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),801). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),801).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),802). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),802).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),803). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),803).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),804). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),804).
:- holdsAt(meeting(wk2gt_1,wk2gt_2),805). :- holdsAt(meeting(wk2gt_2,wk2gt_1),805).
:- simpleEvent(T,P,walking), T>780 , T<804.
:- simpleEvent(T,P,running), T>780 , T<804.

#show simpleEvent/3.