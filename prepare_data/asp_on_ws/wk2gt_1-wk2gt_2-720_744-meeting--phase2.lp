

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
orientation(wk2gt_1,125,720). coords(wk2gt_1,114,53,720).  visible(wk2gt_1,720). orientation(wk2gt_2,142,720). coords(wk2gt_2,99,63,720).  visible(wk2gt_2,720).
orientation(wk2gt_1,125,721). coords(wk2gt_1,114,53,721).  visible(wk2gt_1,721). orientation(wk2gt_2,142,721). coords(wk2gt_2,99,63,721).  visible(wk2gt_2,721).
orientation(wk2gt_1,125,722). coords(wk2gt_1,114,53,722).  visible(wk2gt_1,722). orientation(wk2gt_2,142,722). coords(wk2gt_2,99,63,722).  visible(wk2gt_2,722).
orientation(wk2gt_1,125,723). coords(wk2gt_1,114,53,723).  visible(wk2gt_1,723). orientation(wk2gt_2,142,723). coords(wk2gt_2,99,63,723).  visible(wk2gt_2,723).
orientation(wk2gt_1,125,724). coords(wk2gt_1,114,53,724).  visible(wk2gt_1,724). orientation(wk2gt_2,142,724). coords(wk2gt_2,99,63,724).  visible(wk2gt_2,724).
orientation(wk2gt_1,125,725). coords(wk2gt_1,114,53,725).  visible(wk2gt_1,725). orientation(wk2gt_2,142,725). coords(wk2gt_2,99,63,725).  visible(wk2gt_2,725).
orientation(wk2gt_1,125,726). coords(wk2gt_1,114,53,726).  visible(wk2gt_1,726). orientation(wk2gt_2,142,726). coords(wk2gt_2,99,63,726).  visible(wk2gt_2,726).
orientation(wk2gt_1,125,727). coords(wk2gt_1,114,53,727).  visible(wk2gt_1,727). orientation(wk2gt_2,142,727). coords(wk2gt_2,99,63,727).  visible(wk2gt_2,727).
orientation(wk2gt_1,125,728). coords(wk2gt_1,114,53,728).  visible(wk2gt_1,728). orientation(wk2gt_2,142,728). coords(wk2gt_2,99,63,728).  visible(wk2gt_2,728).
orientation(wk2gt_1,125,729). coords(wk2gt_1,114,53,729).  visible(wk2gt_1,729). orientation(wk2gt_2,142,729). coords(wk2gt_2,99,63,729).  visible(wk2gt_2,729).
orientation(wk2gt_1,125,730). coords(wk2gt_1,114,53,730).  visible(wk2gt_1,730). orientation(wk2gt_2,142,730). coords(wk2gt_2,99,63,730).  visible(wk2gt_2,730).
orientation(wk2gt_1,125,731). coords(wk2gt_1,114,53,731).  visible(wk2gt_1,731). orientation(wk2gt_2,142,731). coords(wk2gt_2,99,63,731).  visible(wk2gt_2,731).
orientation(wk2gt_1,125,732). coords(wk2gt_1,114,53,732).  visible(wk2gt_1,732). orientation(wk2gt_2,142,732). coords(wk2gt_2,99,63,732).  visible(wk2gt_2,732).
orientation(wk2gt_1,125,733). coords(wk2gt_1,114,53,733).  visible(wk2gt_1,733). orientation(wk2gt_2,142,733). coords(wk2gt_2,99,63,733).  visible(wk2gt_2,733).
orientation(wk2gt_1,125,734). coords(wk2gt_1,114,53,734).  visible(wk2gt_1,734). orientation(wk2gt_2,142,734). coords(wk2gt_2,99,63,734).  visible(wk2gt_2,734).
orientation(wk2gt_1,125,735). coords(wk2gt_1,114,53,735).  visible(wk2gt_1,735). orientation(wk2gt_2,142,735). coords(wk2gt_2,99,63,735).  visible(wk2gt_2,735).
orientation(wk2gt_1,125,736). coords(wk2gt_1,114,53,736).  visible(wk2gt_1,736). orientation(wk2gt_2,142,736). coords(wk2gt_2,99,63,736).  visible(wk2gt_2,736).
orientation(wk2gt_1,125,737). coords(wk2gt_1,114,53,737).  visible(wk2gt_1,737). orientation(wk2gt_2,142,737). coords(wk2gt_2,99,63,737).  visible(wk2gt_2,737).
orientation(wk2gt_1,125,738). coords(wk2gt_1,114,53,738).  visible(wk2gt_1,738). orientation(wk2gt_2,142,738). coords(wk2gt_2,99,63,738).  visible(wk2gt_2,738).
orientation(wk2gt_1,125,739). coords(wk2gt_1,114,53,739).  visible(wk2gt_1,739). orientation(wk2gt_2,142,739). coords(wk2gt_2,99,63,739).  visible(wk2gt_2,739).
orientation(wk2gt_1,125,740). coords(wk2gt_1,114,53,740).  visible(wk2gt_1,740). orientation(wk2gt_2,142,740). coords(wk2gt_2,99,63,740).  visible(wk2gt_2,740).
orientation(wk2gt_1,125,741). coords(wk2gt_1,114,53,741).  visible(wk2gt_1,741). orientation(wk2gt_2,142,741). coords(wk2gt_2,99,63,741).  visible(wk2gt_2,741).
orientation(wk2gt_1,125,742). coords(wk2gt_1,114,53,742).  visible(wk2gt_1,742). orientation(wk2gt_2,142,742). coords(wk2gt_2,99,63,742).  visible(wk2gt_2,742).
orientation(wk2gt_1,125,743). coords(wk2gt_1,114,53,743).  visible(wk2gt_1,743). orientation(wk2gt_2,142,743). coords(wk2gt_2,99,63,743).  visible(wk2gt_2,743).
orientation(wk2gt_1,125,744). coords(wk2gt_1,114,53,744).  visible(wk2gt_1,744). orientation(wk2gt_2,142,744). coords(wk2gt_2,99,63,744).  visible(wk2gt_2,744).
orientation(wk2gt_1,125,745). coords(wk2gt_1,114,53,745).  visible(wk2gt_1,745). orientation(wk2gt_2,142,745). coords(wk2gt_2,99,63,745).  visible(wk2gt_2,745).
happensAt(disappear(wk2gt_1),744). happensAt(disappear(wk2gt_2),744).
1{simpleEvent(T,wk2gt_1,active); simpleEvent(T,wk2gt_1,inactive); simpleEvent(T,wk2gt_1,walking); simpleEvent(T,wk2gt_1,running)}1 :- T=720..744.
1{simpleEvent(T,wk2gt_2,active); simpleEvent(T,wk2gt_2,inactive); simpleEvent(T,wk2gt_2,walking); simpleEvent(T,wk2gt_2,running)}1 :- T=720..744.
:- holdsAt(meeting(wk2gt_1,wk2gt_2),720). :- holdsAt(meeting(wk2gt_2,wk2gt_1),720).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),721). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),721).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),722). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),722).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),723). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),723).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),724). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),724).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),725). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),725).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),726). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),726).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),727). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),727).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),728). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),728).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),729). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),729).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),730). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),730).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),731). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),731).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),732). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),732).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),733). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),733).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),734). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),734).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),735). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),735).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),736). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),736).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),737). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),737).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),738). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),738).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),739). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),739).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),740). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),740).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),741). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),741).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),742). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),742).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),743). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),743).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),744). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),744).
:- holdsAt(meeting(wk2gt_1,wk2gt_2),745). :- holdsAt(meeting(wk2gt_2,wk2gt_1),745).
:- simpleEvent(T,P,walking), T>720 , T<744.
:- simpleEvent(T,P,running), T>720 , T<744.

#show simpleEvent/3.