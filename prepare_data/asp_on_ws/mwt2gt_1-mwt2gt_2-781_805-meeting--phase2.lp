

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

	person(mwt2gt_1;mwt2gt_2).
orientation(mwt2gt_1,153,781). coords(mwt2gt_1,30,119,781).  visible(mwt2gt_1,781). orientation(mwt2gt_2,152,781). coords(mwt2gt_2,28,125,781).  visible(mwt2gt_2,781).
orientation(mwt2gt_1,153,782). coords(mwt2gt_1,30,119,782).  visible(mwt2gt_1,782). orientation(mwt2gt_2,152,782). coords(mwt2gt_2,28,125,782).  visible(mwt2gt_2,782).
orientation(mwt2gt_1,153,783). coords(mwt2gt_1,30,119,783).  visible(mwt2gt_1,783). orientation(mwt2gt_2,152,783). coords(mwt2gt_2,28,125,783).  visible(mwt2gt_2,783).
orientation(mwt2gt_1,153,784). coords(mwt2gt_1,30,119,784).  visible(mwt2gt_1,784). orientation(mwt2gt_2,152,784). coords(mwt2gt_2,28,125,784).  visible(mwt2gt_2,784).
orientation(mwt2gt_1,153,785). coords(mwt2gt_1,30,119,785).  visible(mwt2gt_1,785). orientation(mwt2gt_2,152,785). coords(mwt2gt_2,28,125,785).  visible(mwt2gt_2,785).
orientation(mwt2gt_1,153,786). coords(mwt2gt_1,30,119,786).  visible(mwt2gt_1,786). orientation(mwt2gt_2,152,786). coords(mwt2gt_2,28,125,786).  visible(mwt2gt_2,786).
orientation(mwt2gt_1,153,787). coords(mwt2gt_1,30,121,787).  visible(mwt2gt_1,787). orientation(mwt2gt_2,152,787). coords(mwt2gt_2,28,125,787).  visible(mwt2gt_2,787).
orientation(mwt2gt_1,153,788). coords(mwt2gt_1,30,121,788).  visible(mwt2gt_1,788). orientation(mwt2gt_2,152,788). coords(mwt2gt_2,28,126,788).  visible(mwt2gt_2,788).
orientation(mwt2gt_1,153,789). coords(mwt2gt_1,30,121,789).  visible(mwt2gt_1,789). orientation(mwt2gt_2,152,789). coords(mwt2gt_2,28,126,789).  visible(mwt2gt_2,789).
orientation(mwt2gt_1,153,790). coords(mwt2gt_1,30,121,790).  visible(mwt2gt_1,790). orientation(mwt2gt_2,152,790). coords(mwt2gt_2,28,126,790).  visible(mwt2gt_2,790).
orientation(mwt2gt_1,153,791). coords(mwt2gt_1,30,121,791).  visible(mwt2gt_1,791). orientation(mwt2gt_2,152,791). coords(mwt2gt_2,28,126,791).  visible(mwt2gt_2,791).
orientation(mwt2gt_1,153,792). coords(mwt2gt_1,30,122,792).  visible(mwt2gt_1,792). orientation(mwt2gt_2,152,792). coords(mwt2gt_2,28,127,792).  visible(mwt2gt_2,792).
orientation(mwt2gt_1,153,793). coords(mwt2gt_1,30,122,793).  visible(mwt2gt_1,793). orientation(mwt2gt_2,152,793). coords(mwt2gt_2,28,125,793).  visible(mwt2gt_2,793).
orientation(mwt2gt_1,153,794). coords(mwt2gt_1,30,122,794).  visible(mwt2gt_1,794). orientation(mwt2gt_2,152,794). coords(mwt2gt_2,28,125,794).  visible(mwt2gt_2,794).
orientation(mwt2gt_1,153,795). coords(mwt2gt_1,30,122,795).  visible(mwt2gt_1,795). orientation(mwt2gt_2,152,795). coords(mwt2gt_2,28,125,795).  visible(mwt2gt_2,795).
orientation(mwt2gt_1,153,796). coords(mwt2gt_1,30,122,796).  visible(mwt2gt_1,796). orientation(mwt2gt_2,152,796). coords(mwt2gt_2,28,125,796).  visible(mwt2gt_2,796).
orientation(mwt2gt_1,153,797). coords(mwt2gt_1,30,122,797).  visible(mwt2gt_1,797). orientation(mwt2gt_2,152,797). coords(mwt2gt_2,28,125,797).  visible(mwt2gt_2,797).
orientation(mwt2gt_1,153,798). coords(mwt2gt_1,30,122,798).  visible(mwt2gt_1,798). orientation(mwt2gt_2,152,798). coords(mwt2gt_2,28,125,798).  visible(mwt2gt_2,798).
orientation(mwt2gt_1,153,799). coords(mwt2gt_1,30,122,799).  visible(mwt2gt_1,799). orientation(mwt2gt_2,152,799). coords(mwt2gt_2,28,125,799).  visible(mwt2gt_2,799).
orientation(mwt2gt_1,153,800). coords(mwt2gt_1,30,122,800).  visible(mwt2gt_1,800). orientation(mwt2gt_2,152,800). coords(mwt2gt_2,28,125,800).  visible(mwt2gt_2,800).
orientation(mwt2gt_1,153,801). coords(mwt2gt_1,30,122,801).  visible(mwt2gt_1,801). orientation(mwt2gt_2,152,801). coords(mwt2gt_2,28,125,801).  visible(mwt2gt_2,801).
orientation(mwt2gt_1,153,802). coords(mwt2gt_1,30,122,802).  visible(mwt2gt_1,802). orientation(mwt2gt_2,152,802). coords(mwt2gt_2,28,125,802).  visible(mwt2gt_2,802).
orientation(mwt2gt_1,153,803). coords(mwt2gt_1,30,122,803).  visible(mwt2gt_1,803). orientation(mwt2gt_2,152,803). coords(mwt2gt_2,28,125,803).  visible(mwt2gt_2,803).
orientation(mwt2gt_1,153,804). coords(mwt2gt_1,30,122,804).  visible(mwt2gt_1,804). orientation(mwt2gt_2,152,804). coords(mwt2gt_2,28,125,804).  visible(mwt2gt_2,804).
orientation(mwt2gt_1,153,805). coords(mwt2gt_1,30,122,805).  visible(mwt2gt_1,805). orientation(mwt2gt_2,152,805). coords(mwt2gt_2,28,125,805).  visible(mwt2gt_2,805).
orientation(mwt2gt_1,153,806). coords(mwt2gt_1,30,122,806).  visible(mwt2gt_1,806). orientation(mwt2gt_2,152,806). coords(mwt2gt_2,28,125,806).  visible(mwt2gt_2,806).
happensAt(disappear(mwt2gt_1),805). happensAt(disappear(mwt2gt_2),805).
1{simpleEvent(T,mwt2gt_1,active); simpleEvent(T,mwt2gt_1,inactive); simpleEvent(T,mwt2gt_1,walking); simpleEvent(T,mwt2gt_1,running)}1 :- T=781..805.
1{simpleEvent(T,mwt2gt_2,active); simpleEvent(T,mwt2gt_2,inactive); simpleEvent(T,mwt2gt_2,walking); simpleEvent(T,mwt2gt_2,running)}1 :- T=781..805.
:- holdsAt(meeting(mwt2gt_1,mwt2gt_2),781). :- holdsAt(meeting(mwt2gt_2,mwt2gt_1),781).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),782). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),782).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),783). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),783).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),784). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),784).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),785). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),785).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),786). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),786).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),787). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),787).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),788). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),788).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),789). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),789).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),790). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),790).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),791). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),791).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),792). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),792).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),793). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),793).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),794). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),794).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),795). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),795).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),796). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),796).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),797). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),797).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),798). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),798).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),799). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),799).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),800). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),800).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),801). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),801).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),802). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),802).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),803). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),803).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),804). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),804).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),805). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),805).
:- holdsAt(meeting(mwt2gt_1,mwt2gt_2),806). :- holdsAt(meeting(mwt2gt_2,mwt2gt_1),806).
:- simpleEvent(T,P,walking), T>781 , T<805.
:- simpleEvent(T,P,running), T>781 , T<805.

#show simpleEvent/3.