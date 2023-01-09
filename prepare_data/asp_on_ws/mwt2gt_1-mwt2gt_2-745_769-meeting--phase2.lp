

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
orientation(mwt2gt_1,153,745). coords(mwt2gt_1,30,121,745).  visible(mwt2gt_1,745). orientation(mwt2gt_2,152,745). coords(mwt2gt_2,27,128,745).  visible(mwt2gt_2,745).
orientation(mwt2gt_1,153,746). coords(mwt2gt_1,30,121,746).  visible(mwt2gt_1,746). orientation(mwt2gt_2,152,746). coords(mwt2gt_2,27,128,746).  visible(mwt2gt_2,746).
orientation(mwt2gt_1,153,747). coords(mwt2gt_1,30,121,747).  visible(mwt2gt_1,747). orientation(mwt2gt_2,152,747). coords(mwt2gt_2,27,128,747).  visible(mwt2gt_2,747).
orientation(mwt2gt_1,153,748). coords(mwt2gt_1,30,121,748).  visible(mwt2gt_1,748). orientation(mwt2gt_2,152,748). coords(mwt2gt_2,27,128,748).  visible(mwt2gt_2,748).
orientation(mwt2gt_1,153,749). coords(mwt2gt_1,30,121,749).  visible(mwt2gt_1,749). orientation(mwt2gt_2,152,749). coords(mwt2gt_2,27,128,749).  visible(mwt2gt_2,749).
orientation(mwt2gt_1,153,750). coords(mwt2gt_1,30,121,750).  visible(mwt2gt_1,750). orientation(mwt2gt_2,152,750). coords(mwt2gt_2,27,128,750).  visible(mwt2gt_2,750).
orientation(mwt2gt_1,153,751). coords(mwt2gt_1,30,121,751).  visible(mwt2gt_1,751). orientation(mwt2gt_2,152,751). coords(mwt2gt_2,28,128,751).  visible(mwt2gt_2,751).
orientation(mwt2gt_1,153,752). coords(mwt2gt_1,30,121,752).  visible(mwt2gt_1,752). orientation(mwt2gt_2,152,752). coords(mwt2gt_2,28,128,752).  visible(mwt2gt_2,752).
orientation(mwt2gt_1,153,753). coords(mwt2gt_1,30,120,753).  visible(mwt2gt_1,753). orientation(mwt2gt_2,152,753). coords(mwt2gt_2,28,128,753).  visible(mwt2gt_2,753).
orientation(mwt2gt_1,153,754). coords(mwt2gt_1,30,120,754).  visible(mwt2gt_1,754). orientation(mwt2gt_2,152,754). coords(mwt2gt_2,28,128,754).  visible(mwt2gt_2,754).
orientation(mwt2gt_1,153,755). coords(mwt2gt_1,30,120,755).  visible(mwt2gt_1,755). orientation(mwt2gt_2,152,755). coords(mwt2gt_2,28,128,755).  visible(mwt2gt_2,755).
orientation(mwt2gt_1,153,756). coords(mwt2gt_1,30,120,756).  visible(mwt2gt_1,756). orientation(mwt2gt_2,152,756). coords(mwt2gt_2,28,128,756).  visible(mwt2gt_2,756).
orientation(mwt2gt_1,153,757). coords(mwt2gt_1,30,120,757).  visible(mwt2gt_1,757). orientation(mwt2gt_2,152,757). coords(mwt2gt_2,28,128,757).  visible(mwt2gt_2,757).
orientation(mwt2gt_1,153,758). coords(mwt2gt_1,30,120,758).  visible(mwt2gt_1,758). orientation(mwt2gt_2,152,758). coords(mwt2gt_2,28,128,758).  visible(mwt2gt_2,758).
orientation(mwt2gt_1,153,759). coords(mwt2gt_1,30,120,759).  visible(mwt2gt_1,759). orientation(mwt2gt_2,152,759). coords(mwt2gt_2,28,128,759).  visible(mwt2gt_2,759).
orientation(mwt2gt_1,153,760). coords(mwt2gt_1,30,120,760).  visible(mwt2gt_1,760). orientation(mwt2gt_2,152,760). coords(mwt2gt_2,28,128,760).  visible(mwt2gt_2,760).
orientation(mwt2gt_1,153,761). coords(mwt2gt_1,30,120,761).  visible(mwt2gt_1,761). orientation(mwt2gt_2,152,761). coords(mwt2gt_2,28,128,761).  visible(mwt2gt_2,761).
orientation(mwt2gt_1,153,762). coords(mwt2gt_1,30,120,762).  visible(mwt2gt_1,762). orientation(mwt2gt_2,152,762). coords(mwt2gt_2,28,128,762).  visible(mwt2gt_2,762).
orientation(mwt2gt_1,153,763). coords(mwt2gt_1,30,119,763).  visible(mwt2gt_1,763). orientation(mwt2gt_2,152,763). coords(mwt2gt_2,28,128,763).  visible(mwt2gt_2,763).
orientation(mwt2gt_1,153,764). coords(mwt2gt_1,30,119,764).  visible(mwt2gt_1,764). orientation(mwt2gt_2,152,764). coords(mwt2gt_2,28,128,764).  visible(mwt2gt_2,764).
orientation(mwt2gt_1,153,765). coords(mwt2gt_1,30,119,765).  visible(mwt2gt_1,765). orientation(mwt2gt_2,152,765). coords(mwt2gt_2,28,125,765).  visible(mwt2gt_2,765).
orientation(mwt2gt_1,153,766). coords(mwt2gt_1,30,119,766).  visible(mwt2gt_1,766). orientation(mwt2gt_2,152,766). coords(mwt2gt_2,28,125,766).  visible(mwt2gt_2,766).
orientation(mwt2gt_1,153,767). coords(mwt2gt_1,30,119,767).  visible(mwt2gt_1,767). orientation(mwt2gt_2,152,767). coords(mwt2gt_2,28,125,767).  visible(mwt2gt_2,767).
orientation(mwt2gt_1,153,768). coords(mwt2gt_1,30,119,768).  visible(mwt2gt_1,768). orientation(mwt2gt_2,152,768). coords(mwt2gt_2,28,125,768).  visible(mwt2gt_2,768).
orientation(mwt2gt_1,153,769). coords(mwt2gt_1,30,119,769).  visible(mwt2gt_1,769). orientation(mwt2gt_2,152,769). coords(mwt2gt_2,28,125,769).  visible(mwt2gt_2,769).
orientation(mwt2gt_1,153,770). coords(mwt2gt_1,30,119,770).  visible(mwt2gt_1,770). orientation(mwt2gt_2,152,770). coords(mwt2gt_2,28,125,770).  visible(mwt2gt_2,770).
happensAt(disappear(mwt2gt_1),769). happensAt(disappear(mwt2gt_2),769).
1{simpleEvent(T,mwt2gt_1,active); simpleEvent(T,mwt2gt_1,inactive); simpleEvent(T,mwt2gt_1,walking); simpleEvent(T,mwt2gt_1,running)}1 :- T=745..769.
1{simpleEvent(T,mwt2gt_2,active); simpleEvent(T,mwt2gt_2,inactive); simpleEvent(T,mwt2gt_2,walking); simpleEvent(T,mwt2gt_2,running)}1 :- T=745..769.
:- holdsAt(meeting(mwt2gt_1,mwt2gt_2),745). :- holdsAt(meeting(mwt2gt_2,mwt2gt_1),745).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),746). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),746).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),747). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),747).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),748). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),748).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),749). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),749).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),750). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),750).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),751). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),751).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),752). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),752).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),753). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),753).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),754). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),754).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),755). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),755).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),756). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),756).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),757). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),757).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),758). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),758).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),759). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),759).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),760). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),760).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),761). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),761).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),762). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),762).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),763). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),763).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),764). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),764).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),765). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),765).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),766). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),766).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),767). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),767).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),768). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),768).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),769). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),769).
:- holdsAt(meeting(mwt2gt_1,mwt2gt_2),770). :- holdsAt(meeting(mwt2gt_2,mwt2gt_1),770).
:- simpleEvent(T,P,walking), T>745 , T<769.
:- simpleEvent(T,P,running), T>745 , T<769.

#show simpleEvent/3.