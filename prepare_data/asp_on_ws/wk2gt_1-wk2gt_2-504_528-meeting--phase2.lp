

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
orientation(wk2gt_1,122,504). coords(wk2gt_1,115,53,504).  visible(wk2gt_1,504). orientation(wk2gt_2,141,504). coords(wk2gt_2,99,63,504).  visible(wk2gt_2,504).
orientation(wk2gt_1,122,505). coords(wk2gt_1,115,53,505).  visible(wk2gt_1,505). orientation(wk2gt_2,141,505). coords(wk2gt_2,99,63,505).  visible(wk2gt_2,505).
orientation(wk2gt_1,122,506). coords(wk2gt_1,115,53,506).  visible(wk2gt_1,506). orientation(wk2gt_2,141,506). coords(wk2gt_2,99,63,506).  visible(wk2gt_2,506).
orientation(wk2gt_1,122,507). coords(wk2gt_1,115,53,507).  visible(wk2gt_1,507). orientation(wk2gt_2,141,507). coords(wk2gt_2,99,63,507).  visible(wk2gt_2,507).
orientation(wk2gt_1,122,508). coords(wk2gt_1,115,53,508).  visible(wk2gt_1,508). orientation(wk2gt_2,141,508). coords(wk2gt_2,99,63,508).  visible(wk2gt_2,508).
orientation(wk2gt_1,122,509). coords(wk2gt_1,115,53,509).  visible(wk2gt_1,509). orientation(wk2gt_2,141,509). coords(wk2gt_2,99,63,509).  visible(wk2gt_2,509).
orientation(wk2gt_1,122,510). coords(wk2gt_1,115,53,510).  visible(wk2gt_1,510). orientation(wk2gt_2,141,510). coords(wk2gt_2,99,63,510).  visible(wk2gt_2,510).
orientation(wk2gt_1,122,511). coords(wk2gt_1,115,53,511).  visible(wk2gt_1,511). orientation(wk2gt_2,141,511). coords(wk2gt_2,99,63,511).  visible(wk2gt_2,511).
orientation(wk2gt_1,122,512). coords(wk2gt_1,115,53,512).  visible(wk2gt_1,512). orientation(wk2gt_2,141,512). coords(wk2gt_2,99,63,512).  visible(wk2gt_2,512).
orientation(wk2gt_1,122,513). coords(wk2gt_1,115,53,513).  visible(wk2gt_1,513). orientation(wk2gt_2,141,513). coords(wk2gt_2,99,63,513).  visible(wk2gt_2,513).
orientation(wk2gt_1,122,514). coords(wk2gt_1,115,53,514).  visible(wk2gt_1,514). orientation(wk2gt_2,141,514). coords(wk2gt_2,99,63,514).  visible(wk2gt_2,514).
orientation(wk2gt_1,122,515). coords(wk2gt_1,115,53,515).  visible(wk2gt_1,515). orientation(wk2gt_2,141,515). coords(wk2gt_2,99,63,515).  visible(wk2gt_2,515).
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
happensAt(disappear(wk2gt_1),528). happensAt(disappear(wk2gt_2),528).
1{simpleEvent(T,wk2gt_1,active); simpleEvent(T,wk2gt_1,inactive); simpleEvent(T,wk2gt_1,walking); simpleEvent(T,wk2gt_1,running)}1 :- T=504..528.
1{simpleEvent(T,wk2gt_2,active); simpleEvent(T,wk2gt_2,inactive); simpleEvent(T,wk2gt_2,walking); simpleEvent(T,wk2gt_2,running)}1 :- T=504..528.
:- holdsAt(meeting(wk2gt_1,wk2gt_2),504). :- holdsAt(meeting(wk2gt_2,wk2gt_1),504).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),505). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),505).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),506). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),506).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),507). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),507).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),508). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),508).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),509). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),509).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),510). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),510).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),511). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),511).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),512). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),512).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),513). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),513).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),514). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),514).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),515). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),515).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),516). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),516).
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
:- holdsAt(meeting(wk2gt_1,wk2gt_2),529). :- holdsAt(meeting(wk2gt_2,wk2gt_1),529).
:- simpleEvent(T,P,walking), T>504 , T<528.
:- simpleEvent(T,P,running), T>504 , T<528.

#show simpleEvent/3.