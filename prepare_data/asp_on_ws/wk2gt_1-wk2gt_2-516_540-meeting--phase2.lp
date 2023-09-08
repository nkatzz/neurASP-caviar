

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
orientation(wk2gt_1,122,516). coords(wk2gt_1,115,53,516).  visible(wk2gt_1,516). orientation(wk2gt_2,141,516). coords(wk2gt_2,99,63,516).  visible(wk2gt_2,516).
orientation(wk2gt_1,122,517). coords(wk2gt_1,115,53,517).  visible(wk2gt_1,517). orientation(wk2gt_2,141,517). coords(wk2gt_2,99,63,517).  visible(wk2gt_2,517).
orientation(wk2gt_1,122,518). coords(wk2gt_1,115,53,518).  visible(wk2gt_1,518). orientation(wk2gt_2,141,518). coords(wk2gt_2,99,63,518).  visible(wk2gt_2,518).
orientation(wk2gt_1,122,519). coords(wk2gt_1,115,53,519).  visible(wk2gt_1,519). orientation(wk2gt_2,141,519). coords(wk2gt_2,99,63,519).  visible(wk2gt_2,519).
orientation(wk2gt_1,122,520). coords(wk2gt_1,115,53,520).  visible(wk2gt_1,520). orientation(wk2gt_2,141,520). coords(wk2gt_2,99,63,520).  visible(wk2gt_2,520).
orientation(wk2gt_1,122,521). coords(wk2gt_1,115,53,521).  visible(wk2gt_1,521). orientation(wk2gt_2,141,521). coords(wk2gt_2,99,63,521).  visible(wk2gt_2,521).
orientation(wk2gt_1,122,522). coords(wk2gt_1,115,53,522).  visible(wk2gt_1,522). orientation(wk2gt_2,141,522). coords(wk2gt_2,99,63,522).  visible(wk2gt_2,522).
orientation(wk2gt_1,122,523). coords(wk2gt_1,115,53,523).  visible(wk2gt_1,523). orientation(wk2gt_2,141,523). coords(wk2gt_2,99,63,523).  visible(wk2gt_2,523).
orientation(wk2gt_1,122,524). coords(wk2gt_1,115,53,524).  visible(wk2gt_1,524). orientation(wk2gt_2,141,524). coords(wk2gt_2,99,63,524).  visible(wk2gt_2,524).
orientation(wk2gt_1,122,525). coords(wk2gt_1,115,53,525).  visible(wk2gt_1,525). orientation(wk2gt_2,141,525). coords(wk2gt_2,99,63,525).  visible(wk2gt_2,525).
orientation(wk2gt_1,122,526). coords(wk2gt_1,115,53,526).  visible(wk2gt_1,526). orientation(wk2gt_2,141,526). coords(wk2gt_2,99,63,526).  visible(wk2gt_2,526).
orientation(wk2gt_1,122,527). coords(wk2gt_1,115,53,527).  visible(wk2gt_1,527). orientation(wk2gt_2,141,527). coords(wk2gt_2,99,63,527).  visible(wk2gt_2,527).
orientation(wk2gt_1,122,528). coords(wk2gt_1,115,53,528).  visible(wk2gt_1,528). orientation(wk2gt_2,141,528). coords(wk2gt_2,99,63,528).  visible(wk2gt_2,528).
orientation(wk2gt_1,122,529). coords(wk2gt_1,115,53,529).  visible(wk2gt_1,529). orientation(wk2gt_2,141,529). coords(wk2gt_2,99,63,529).  visible(wk2gt_2,529).
orientation(wk2gt_1,122,530). coords(wk2gt_1,115,53,530).  visible(wk2gt_1,530). orientation(wk2gt_2,141,530). coords(wk2gt_2,99,63,530).  visible(wk2gt_2,530).
orientation(wk2gt_1,122,531). coords(wk2gt_1,115,53,531).  visible(wk2gt_1,531). orientation(wk2gt_2,141,531). coords(wk2gt_2,99,63,531).  visible(wk2gt_2,531).
orientation(wk2gt_1,122,532). coords(wk2gt_1,115,53,532).  visible(wk2gt_1,532). orientation(wk2gt_2,141,532). coords(wk2gt_2,99,63,532).  visible(wk2gt_2,532).
orientation(wk2gt_1,122,533). coords(wk2gt_1,115,53,533).  visible(wk2gt_1,533). orientation(wk2gt_2,141,533). coords(wk2gt_2,99,63,533).  visible(wk2gt_2,533).
orientation(wk2gt_1,122,534). coords(wk2gt_1,115,53,534).  visible(wk2gt_1,534). orientation(wk2gt_2,141,534). coords(wk2gt_2,99,63,534).  visible(wk2gt_2,534).
orientation(wk2gt_1,122,535). coords(wk2gt_1,115,53,535).  visible(wk2gt_1,535). orientation(wk2gt_2,141,535). coords(wk2gt_2,99,63,535).  visible(wk2gt_2,535).
orientation(wk2gt_1,122,536). coords(wk2gt_1,115,53,536).  visible(wk2gt_1,536). orientation(wk2gt_2,141,536). coords(wk2gt_2,99,63,536).  visible(wk2gt_2,536).
orientation(wk2gt_1,122,537). coords(wk2gt_1,115,53,537).  visible(wk2gt_1,537). orientation(wk2gt_2,141,537). coords(wk2gt_2,99,63,537).  visible(wk2gt_2,537).
orientation(wk2gt_1,122,538). coords(wk2gt_1,115,53,538).  visible(wk2gt_1,538). orientation(wk2gt_2,141,538). coords(wk2gt_2,99,63,538).  visible(wk2gt_2,538).
orientation(wk2gt_1,122,539). coords(wk2gt_1,115,53,539).  visible(wk2gt_1,539). orientation(wk2gt_2,141,539). coords(wk2gt_2,99,63,539).  visible(wk2gt_2,539).
orientation(wk2gt_1,122,540). coords(wk2gt_1,115,53,540).  visible(wk2gt_1,540). orientation(wk2gt_2,141,540). coords(wk2gt_2,99,63,540).  visible(wk2gt_2,540).
orientation(wk2gt_1,122,541). coords(wk2gt_1,115,53,541).  visible(wk2gt_1,541). orientation(wk2gt_2,141,541). coords(wk2gt_2,99,63,541).  visible(wk2gt_2,541).
happensAt(disappear(wk2gt_1),540). happensAt(disappear(wk2gt_2),540).
1{simpleEvent(T,wk2gt_1,active); simpleEvent(T,wk2gt_1,inactive); simpleEvent(T,wk2gt_1,walking); simpleEvent(T,wk2gt_1,running)}1 :- T=516..540.
1{simpleEvent(T,wk2gt_2,active); simpleEvent(T,wk2gt_2,inactive); simpleEvent(T,wk2gt_2,walking); simpleEvent(T,wk2gt_2,running)}1 :- T=516..540.
:- holdsAt(meeting(wk2gt_1,wk2gt_2),516). :- holdsAt(meeting(wk2gt_2,wk2gt_1),516).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),517). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),517).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),518). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),518).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),519). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),519).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),520). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),520).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),521). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),521).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),522). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),522).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),523). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),523).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),524). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),524).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),525). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),525).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),526). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),526).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),527). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),527).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),528). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),528).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),529). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),529).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),530). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),530).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),531). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),531).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),532). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),532).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),533). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),533).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),534). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),534).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),535). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),535).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),536). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),536).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),537). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),537).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),538). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),538).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),539). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),539).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),540). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),540).
:- holdsAt(meeting(wk2gt_1,wk2gt_2),541). :- holdsAt(meeting(wk2gt_2,wk2gt_1),541).
:- simpleEvent(T,P,walking), T>516 , T<540.
:- simpleEvent(T,P,running), T>516 , T<540.

#show simpleEvent/3.