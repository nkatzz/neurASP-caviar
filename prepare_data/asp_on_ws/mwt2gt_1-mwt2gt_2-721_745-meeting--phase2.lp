

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
orientation(mwt2gt_1,153,721). coords(mwt2gt_1,27,126,721).  visible(mwt2gt_1,721). orientation(mwt2gt_2,152,721). coords(mwt2gt_2,24,134,721).  visible(mwt2gt_2,721).
orientation(mwt2gt_1,153,722). coords(mwt2gt_1,26,129,722).  visible(mwt2gt_1,722). orientation(mwt2gt_2,152,722). coords(mwt2gt_2,24,134,722).  visible(mwt2gt_2,722).
orientation(mwt2gt_1,153,723). coords(mwt2gt_1,26,129,723).  visible(mwt2gt_1,723). orientation(mwt2gt_2,152,723). coords(mwt2gt_2,24,134,723).  visible(mwt2gt_2,723).
orientation(mwt2gt_1,153,724). coords(mwt2gt_1,29,126,724).  visible(mwt2gt_1,724). orientation(mwt2gt_2,152,724). coords(mwt2gt_2,26,133,724).  visible(mwt2gt_2,724).
orientation(mwt2gt_1,153,725). coords(mwt2gt_1,29,126,725).  visible(mwt2gt_1,725). orientation(mwt2gt_2,152,725). coords(mwt2gt_2,26,133,725).  visible(mwt2gt_2,725).
orientation(mwt2gt_1,153,726). coords(mwt2gt_1,29,126,726).  visible(mwt2gt_1,726). orientation(mwt2gt_2,152,726). coords(mwt2gt_2,26,133,726).  visible(mwt2gt_2,726).
orientation(mwt2gt_1,153,727). coords(mwt2gt_1,29,126,727).  visible(mwt2gt_1,727). orientation(mwt2gt_2,152,727). coords(mwt2gt_2,26,133,727).  visible(mwt2gt_2,727).
orientation(mwt2gt_1,153,728). coords(mwt2gt_1,29,126,728).  visible(mwt2gt_1,728). orientation(mwt2gt_2,152,728). coords(mwt2gt_2,26,132,728).  visible(mwt2gt_2,728).
orientation(mwt2gt_1,153,729). coords(mwt2gt_1,29,125,729).  visible(mwt2gt_1,729). orientation(mwt2gt_2,152,729). coords(mwt2gt_2,26,132,729).  visible(mwt2gt_2,729).
orientation(mwt2gt_1,153,730). coords(mwt2gt_1,29,125,730).  visible(mwt2gt_1,730). orientation(mwt2gt_2,152,730). coords(mwt2gt_2,26,132,730).  visible(mwt2gt_2,730).
orientation(mwt2gt_1,153,731). coords(mwt2gt_1,30,122,731).  visible(mwt2gt_1,731). orientation(mwt2gt_2,152,731). coords(mwt2gt_2,26,132,731).  visible(mwt2gt_2,731).
orientation(mwt2gt_1,153,732). coords(mwt2gt_1,30,122,732).  visible(mwt2gt_1,732). orientation(mwt2gt_2,152,732). coords(mwt2gt_2,26,130,732).  visible(mwt2gt_2,732).
orientation(mwt2gt_1,153,733). coords(mwt2gt_1,30,122,733).  visible(mwt2gt_1,733). orientation(mwt2gt_2,152,733). coords(mwt2gt_2,26,130,733).  visible(mwt2gt_2,733).
orientation(mwt2gt_1,153,734). coords(mwt2gt_1,30,121,734).  visible(mwt2gt_1,734). orientation(mwt2gt_2,152,734). coords(mwt2gt_2,27,129,734).  visible(mwt2gt_2,734).
orientation(mwt2gt_1,153,735). coords(mwt2gt_1,30,121,735).  visible(mwt2gt_1,735). orientation(mwt2gt_2,152,735). coords(mwt2gt_2,27,129,735).  visible(mwt2gt_2,735).
orientation(mwt2gt_1,153,736). coords(mwt2gt_1,30,121,736).  visible(mwt2gt_1,736). orientation(mwt2gt_2,152,736). coords(mwt2gt_2,27,129,736).  visible(mwt2gt_2,736).
orientation(mwt2gt_1,153,737). coords(mwt2gt_1,30,121,737).  visible(mwt2gt_1,737). orientation(mwt2gt_2,152,737). coords(mwt2gt_2,27,129,737).  visible(mwt2gt_2,737).
orientation(mwt2gt_1,153,738). coords(mwt2gt_1,30,121,738).  visible(mwt2gt_1,738). orientation(mwt2gt_2,152,738). coords(mwt2gt_2,27,129,738).  visible(mwt2gt_2,738).
orientation(mwt2gt_1,153,739). coords(mwt2gt_1,30,119,739).  visible(mwt2gt_1,739). orientation(mwt2gt_2,152,739). coords(mwt2gt_2,27,129,739).  visible(mwt2gt_2,739).
orientation(mwt2gt_1,153,740). coords(mwt2gt_1,30,119,740).  visible(mwt2gt_1,740). orientation(mwt2gt_2,152,740). coords(mwt2gt_2,27,129,740).  visible(mwt2gt_2,740).
orientation(mwt2gt_1,153,741). coords(mwt2gt_1,30,121,741).  visible(mwt2gt_1,741). orientation(mwt2gt_2,152,741). coords(mwt2gt_2,27,129,741).  visible(mwt2gt_2,741).
orientation(mwt2gt_1,153,742). coords(mwt2gt_1,30,119,742).  visible(mwt2gt_1,742). orientation(mwt2gt_2,152,742). coords(mwt2gt_2,27,129,742).  visible(mwt2gt_2,742).
orientation(mwt2gt_1,153,743). coords(mwt2gt_1,30,119,743).  visible(mwt2gt_1,743). orientation(mwt2gt_2,152,743). coords(mwt2gt_2,27,129,743).  visible(mwt2gt_2,743).
orientation(mwt2gt_1,153,744). coords(mwt2gt_1,30,121,744).  visible(mwt2gt_1,744). orientation(mwt2gt_2,152,744). coords(mwt2gt_2,27,129,744).  visible(mwt2gt_2,744).
orientation(mwt2gt_1,153,745). coords(mwt2gt_1,30,121,745).  visible(mwt2gt_1,745). orientation(mwt2gt_2,152,745). coords(mwt2gt_2,27,128,745).  visible(mwt2gt_2,745).
orientation(mwt2gt_1,153,746). coords(mwt2gt_1,30,121,746).  visible(mwt2gt_1,746). orientation(mwt2gt_2,152,746). coords(mwt2gt_2,27,128,746).  visible(mwt2gt_2,746).
happensAt(disappear(mwt2gt_1),745). happensAt(disappear(mwt2gt_2),745).
1{simpleEvent(T,mwt2gt_1,active); simpleEvent(T,mwt2gt_1,inactive); simpleEvent(T,mwt2gt_1,walking); simpleEvent(T,mwt2gt_1,running)}1 :- T=721..745.
1{simpleEvent(T,mwt2gt_2,active); simpleEvent(T,mwt2gt_2,inactive); simpleEvent(T,mwt2gt_2,walking); simpleEvent(T,mwt2gt_2,running)}1 :- T=721..745.
:- holdsAt(meeting(mwt2gt_1,mwt2gt_2),721). :- holdsAt(meeting(mwt2gt_2,mwt2gt_1),721).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),722). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),722).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),723). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),723).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),724). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),724).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),725). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),725).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),726). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),726).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),727). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),727).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),728). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),728).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),729). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),729).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),730). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),730).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),731). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),731).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),732). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),732).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),733). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),733).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),734). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),734).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),735). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),735).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),736). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),736).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),737). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),737).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),738). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),738).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),739). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),739).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),740). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),740).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),741). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),741).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),742). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),742).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),743). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),743).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),744). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),744).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),745). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),745).
:- holdsAt(meeting(mwt2gt_1,mwt2gt_2),746). :- holdsAt(meeting(mwt2gt_2,mwt2gt_1),746).
:- simpleEvent(T,P,walking), T>721 , T<745.
:- simpleEvent(T,P,running), T>721 , T<745.

#show simpleEvent/3.