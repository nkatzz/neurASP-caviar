

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
orientation(wk1gt_4,157,533). coords(wk1gt_4,56,169,533).  visible(wk1gt_4,533). orientation(wk1gt_5,162,533). coords(wk1gt_5,57,150,533).  visible(wk1gt_5,533).
orientation(wk1gt_4,157,534). coords(wk1gt_4,56,169,534).  visible(wk1gt_4,534). orientation(wk1gt_5,162,534). coords(wk1gt_5,57,150,534).  visible(wk1gt_5,534).
orientation(wk1gt_4,157,535). coords(wk1gt_4,56,169,535).  visible(wk1gt_4,535). orientation(wk1gt_5,162,535). coords(wk1gt_5,57,150,535).  visible(wk1gt_5,535).
orientation(wk1gt_4,157,536). coords(wk1gt_4,56,169,536).  visible(wk1gt_4,536). orientation(wk1gt_5,162,536). coords(wk1gt_5,57,150,536).  visible(wk1gt_5,536).
orientation(wk1gt_4,157,537). coords(wk1gt_4,56,169,537).  visible(wk1gt_4,537). orientation(wk1gt_5,162,537). coords(wk1gt_5,57,150,537).  visible(wk1gt_5,537).
orientation(wk1gt_4,157,538). coords(wk1gt_4,56,169,538).  visible(wk1gt_4,538). orientation(wk1gt_5,162,538). coords(wk1gt_5,57,150,538).  visible(wk1gt_5,538).
orientation(wk1gt_4,157,539). coords(wk1gt_4,56,169,539).  visible(wk1gt_4,539). orientation(wk1gt_5,162,539). coords(wk1gt_5,57,150,539).  visible(wk1gt_5,539).
orientation(wk1gt_4,157,540). coords(wk1gt_4,56,169,540).  visible(wk1gt_4,540). orientation(wk1gt_5,162,540). coords(wk1gt_5,57,150,540).  visible(wk1gt_5,540).
orientation(wk1gt_4,157,541). coords(wk1gt_4,56,169,541).  visible(wk1gt_4,541). orientation(wk1gt_5,162,541). coords(wk1gt_5,57,150,541).  visible(wk1gt_5,541).
orientation(wk1gt_4,157,542). coords(wk1gt_4,56,169,542).  visible(wk1gt_4,542). orientation(wk1gt_5,162,542). coords(wk1gt_5,57,150,542).  visible(wk1gt_5,542).
orientation(wk1gt_4,157,543). coords(wk1gt_4,56,169,543).  visible(wk1gt_4,543). orientation(wk1gt_5,162,543). coords(wk1gt_5,57,150,543).  visible(wk1gt_5,543).
orientation(wk1gt_4,157,544). coords(wk1gt_4,56,169,544).  visible(wk1gt_4,544). orientation(wk1gt_5,162,544). coords(wk1gt_5,57,150,544).  visible(wk1gt_5,544).
happensAt(disappear(wk1gt_4),543). happensAt(disappear(wk1gt_5),543).
1{simpleEvent(T,wk1gt_4,active); simpleEvent(T,wk1gt_4,inactive); simpleEvent(T,wk1gt_4,walking); simpleEvent(T,wk1gt_4,running)}1 :- T=519..543.
1{simpleEvent(T,wk1gt_5,active); simpleEvent(T,wk1gt_5,inactive); simpleEvent(T,wk1gt_5,walking); simpleEvent(T,wk1gt_5,running)}1 :- T=519..543.
:- holdsAt(meeting(wk1gt_4,wk1gt_5),519). :- holdsAt(meeting(wk1gt_5,wk1gt_4),519).
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
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),532). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),532).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),533). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),533).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),534). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),534).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),535). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),535).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),536). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),536).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),537). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),537).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),538). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),538).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),539). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),539).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),540). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),540).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),541). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),541).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),542). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),542).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),543). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),543).
:- holdsAt(meeting(wk1gt_4,wk1gt_5),544). :- holdsAt(meeting(wk1gt_5,wk1gt_4),544).
:- simpleEvent(T,P,walking), T>519 , T<543.
:- simpleEvent(T,P,running), T>519 , T<543.

#show simpleEvent/3.