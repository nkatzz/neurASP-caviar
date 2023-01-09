

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

	person(wk1gt_4;wk1gt_5).
orientation(wk1gt_4,157,507). coords(wk1gt_4,56,170,507).  visible(wk1gt_4,507). orientation(wk1gt_5,162,507). coords(wk1gt_5,57,150,507).  visible(wk1gt_5,507).
orientation(wk1gt_4,157,508). coords(wk1gt_4,56,170,508).  visible(wk1gt_4,508). orientation(wk1gt_5,162,508). coords(wk1gt_5,57,150,508).  visible(wk1gt_5,508).
orientation(wk1gt_4,157,509). coords(wk1gt_4,56,170,509).  visible(wk1gt_4,509). orientation(wk1gt_5,162,509). coords(wk1gt_5,57,150,509).  visible(wk1gt_5,509).
orientation(wk1gt_4,157,510). coords(wk1gt_4,56,170,510).  visible(wk1gt_4,510). orientation(wk1gt_5,162,510). coords(wk1gt_5,57,150,510).  visible(wk1gt_5,510).
orientation(wk1gt_4,157,511). coords(wk1gt_4,56,170,511).  visible(wk1gt_4,511). orientation(wk1gt_5,162,511). coords(wk1gt_5,57,150,511).  visible(wk1gt_5,511).
orientation(wk1gt_4,157,512). coords(wk1gt_4,56,170,512).  visible(wk1gt_4,512). orientation(wk1gt_5,162,512). coords(wk1gt_5,57,150,512).  visible(wk1gt_5,512).
orientation(wk1gt_4,157,513). coords(wk1gt_4,56,170,513).  visible(wk1gt_4,513). orientation(wk1gt_5,162,513). coords(wk1gt_5,57,150,513).  visible(wk1gt_5,513).
orientation(wk1gt_4,157,514). coords(wk1gt_4,56,169,514).  visible(wk1gt_4,514). orientation(wk1gt_5,162,514). coords(wk1gt_5,57,150,514).  visible(wk1gt_5,514).
orientation(wk1gt_4,157,515). coords(wk1gt_4,56,169,515).  visible(wk1gt_4,515). orientation(wk1gt_5,162,515). coords(wk1gt_5,57,150,515).  visible(wk1gt_5,515).
orientation(wk1gt_4,157,516). coords(wk1gt_4,56,169,516).  visible(wk1gt_4,516). orientation(wk1gt_5,162,516). coords(wk1gt_5,57,150,516).  visible(wk1gt_5,516).
orientation(wk1gt_4,157,517). coords(wk1gt_4,56,169,517).  visible(wk1gt_4,517). orientation(wk1gt_5,162,517). coords(wk1gt_5,57,150,517).  visible(wk1gt_5,517).
orientation(wk1gt_4,157,518). coords(wk1gt_4,56,169,518).  visible(wk1gt_4,518). orientation(wk1gt_5,162,518). coords(wk1gt_5,57,150,518).  visible(wk1gt_5,518).
orientation(wk1gt_4,157,519). coords(wk1gt_4,56,169,519).  visible(wk1gt_4,519). orientation(wk1gt_5,162,519). coords(wk1gt_5,57,150,519).  visible(wk1gt_5,519).
orientation(wk1gt_4,157,520). coords(wk1gt_4,56,169,520).  visible(wk1gt_4,520). orientation(wk1gt_5,162,520). coords(wk1gt_5,57,150,520).  visible(wk1gt_5,520).
orientation(wk1gt_4,157,521). coords(wk1gt_4,56,169,521).  visible(wk1gt_4,521). orientation(wk1gt_5,162,521). coords(wk1gt_5,57,150,521).  visible(wk1gt_5,521).
orientation(wk1gt_4,157,522). coords(wk1gt_4,56,169,522).  visible(wk1gt_4,522). orientation(wk1gt_5,162,522). coords(wk1gt_5,57,150,522).  visible(wk1gt_5,522).
orientation(wk1gt_4,157,523). coords(wk1gt_4,56,169,523).  visible(wk1gt_4,523). orientation(wk1gt_5,162,523). coords(wk1gt_5,57,150,523).  visible(wk1gt_5,523).
orientation(wk1gt_4,157,524). coords(wk1gt_4,56,169,524).  visible(wk1gt_4,524). orientation(wk1gt_5,162,524). coords(wk1gt_5,57,150,524).  visible(wk1gt_5,524).
orientation(wk1gt_4,157,525). coords(wk1gt_4,56,169,525).  visible(wk1gt_4,525). orientation(wk1gt_5,162,525). coords(wk1gt_5,57,150,525).  visible(wk1gt_5,525).
orientation(wk1gt_4,157,526). coords(wk1gt_4,56,169,526).  visible(wk1gt_4,526). orientation(wk1gt_5,162,526). coords(wk1gt_5,57,150,526).  visible(wk1gt_5,526).
orientation(wk1gt_4,157,527). coords(wk1gt_4,56,169,527).  visible(wk1gt_4,527). orientation(wk1gt_5,162,527). coords(wk1gt_5,57,150,527).  visible(wk1gt_5,527).
orientation(wk1gt_4,157,528). coords(wk1gt_4,56,169,528).  visible(wk1gt_4,528). orientation(wk1gt_5,162,528). coords(wk1gt_5,57,150,528).  visible(wk1gt_5,528).
orientation(wk1gt_4,157,529). coords(wk1gt_4,56,169,529).  visible(wk1gt_4,529). orientation(wk1gt_5,162,529). coords(wk1gt_5,57,150,529).  visible(wk1gt_5,529).
orientation(wk1gt_4,157,530). coords(wk1gt_4,56,169,530).  visible(wk1gt_4,530). orientation(wk1gt_5,162,530). coords(wk1gt_5,57,150,530).  visible(wk1gt_5,530).
orientation(wk1gt_4,157,531). coords(wk1gt_4,56,169,531).  visible(wk1gt_4,531). orientation(wk1gt_5,162,531). coords(wk1gt_5,57,150,531).  visible(wk1gt_5,531).
orientation(wk1gt_4,157,532). coords(wk1gt_4,56,169,532).  visible(wk1gt_4,532). orientation(wk1gt_5,162,532). coords(wk1gt_5,57,150,532).  visible(wk1gt_5,532).
happensAt(disappear(wk1gt_4),531). happensAt(disappear(wk1gt_5),531).
1{simpleEvent(T,wk1gt_4,active); simpleEvent(T,wk1gt_4,inactive); simpleEvent(T,wk1gt_4,walking); simpleEvent(T,wk1gt_4,running)}1 :- T=507..531.
1{simpleEvent(T,wk1gt_5,active); simpleEvent(T,wk1gt_5,inactive); simpleEvent(T,wk1gt_5,walking); simpleEvent(T,wk1gt_5,running)}1 :- T=507..531.
:- holdsAt(meeting(wk1gt_4,wk1gt_5),507). :- holdsAt(meeting(wk1gt_5,wk1gt_4),507).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),508). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),508).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),509). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),509).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),510). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),510).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),511). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),511).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),512). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),512).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),513). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),513).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),514). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),514).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),515). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),515).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),516). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),516).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),517). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),517).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),518). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),518).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),519). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),519).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),520). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),520).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),521). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),521).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),522). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),522).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),523). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),523).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),524). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),524).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),525). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),525).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),526). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),526).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),527). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),527).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),528). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),528).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),529). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),529).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),530). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),530).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),531). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),531).
:- holdsAt(meeting(wk1gt_4,wk1gt_5),532). :- holdsAt(meeting(wk1gt_5,wk1gt_4),532).
:- simpleEvent(T,P,walking), T>518 , T<518.
:- simpleEvent(T,P,running), T>518 , T<518.

#show simpleEvent/3.