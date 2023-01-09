

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
orientation(mwt2gt_1,153,709). coords(mwt2gt_1,26,129,709).  visible(mwt2gt_1,709). orientation(mwt2gt_2,152,709). coords(mwt2gt_2,24,138,709).  visible(mwt2gt_2,709).
orientation(mwt2gt_1,153,710). coords(mwt2gt_1,26,129,710).  visible(mwt2gt_1,710). orientation(mwt2gt_2,152,710). coords(mwt2gt_2,24,138,710).  visible(mwt2gt_2,710).
orientation(mwt2gt_1,153,711). coords(mwt2gt_1,26,129,711).  visible(mwt2gt_1,711). orientation(mwt2gt_2,152,711). coords(mwt2gt_2,24,138,711).  visible(mwt2gt_2,711).
orientation(mwt2gt_1,153,712). coords(mwt2gt_1,26,129,712).  visible(mwt2gt_1,712). orientation(mwt2gt_2,152,712). coords(mwt2gt_2,24,138,712).  visible(mwt2gt_2,712).
orientation(mwt2gt_1,153,713). coords(mwt2gt_1,26,129,713).  visible(mwt2gt_1,713). orientation(mwt2gt_2,152,713). coords(mwt2gt_2,24,138,713).  visible(mwt2gt_2,713).
orientation(mwt2gt_1,153,714). coords(mwt2gt_1,26,129,714).  visible(mwt2gt_1,714). orientation(mwt2gt_2,152,714). coords(mwt2gt_2,24,138,714).  visible(mwt2gt_2,714).
orientation(mwt2gt_1,153,715). coords(mwt2gt_1,26,127,715).  visible(mwt2gt_1,715). orientation(mwt2gt_2,152,715). coords(mwt2gt_2,24,138,715).  visible(mwt2gt_2,715).
orientation(mwt2gt_1,153,716). coords(mwt2gt_1,26,128,716).  visible(mwt2gt_1,716). orientation(mwt2gt_2,152,716). coords(mwt2gt_2,24,138,716).  visible(mwt2gt_2,716).
orientation(mwt2gt_1,153,717). coords(mwt2gt_1,27,127,717).  visible(mwt2gt_1,717). orientation(mwt2gt_2,152,717). coords(mwt2gt_2,24,138,717).  visible(mwt2gt_2,717).
orientation(mwt2gt_1,153,718). coords(mwt2gt_1,27,128,718).  visible(mwt2gt_1,718). orientation(mwt2gt_2,152,718). coords(mwt2gt_2,24,138,718).  visible(mwt2gt_2,718).
orientation(mwt2gt_1,153,719). coords(mwt2gt_1,27,128,719).  visible(mwt2gt_1,719). orientation(mwt2gt_2,152,719). coords(mwt2gt_2,24,138,719).  visible(mwt2gt_2,719).
orientation(mwt2gt_1,153,720). coords(mwt2gt_1,27,127,720).  visible(mwt2gt_1,720). orientation(mwt2gt_2,152,720). coords(mwt2gt_2,24,138,720).  visible(mwt2gt_2,720).
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
orientation(mwt2gt_1,153,734). coords(mwt2gt_1,30,122,734).  visible(mwt2gt_1,734). orientation(mwt2gt_2,152,734). coords(mwt2gt_2,26,130,734).  visible(mwt2gt_2,734).
happensAt(disappear(mwt2gt_1),733). happensAt(disappear(mwt2gt_2),733).
1{simpleEvent(T,mwt2gt_1,active); simpleEvent(T,mwt2gt_1,inactive); simpleEvent(T,mwt2gt_1,walking); simpleEvent(T,mwt2gt_1,running)}1 :- T=709..733.
1{simpleEvent(T,mwt2gt_2,active); simpleEvent(T,mwt2gt_2,inactive); simpleEvent(T,mwt2gt_2,walking); simpleEvent(T,mwt2gt_2,running)}1 :- T=709..733.
:- holdsAt(meeting(mwt2gt_1,mwt2gt_2),709). :- holdsAt(meeting(mwt2gt_2,mwt2gt_1),709).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),710). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),710).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),711). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),711).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),712). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),712).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),713). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),713).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),714). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),714).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),715). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),715).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),716). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),716).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),717). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),717).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),718). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),718).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),719). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),719).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),720). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),720).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),721). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),721).
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
:- holdsAt(meeting(mwt2gt_1,mwt2gt_2),734). :- holdsAt(meeting(mwt2gt_2,mwt2gt_1),734).
:- simpleEvent(T,P,walking), T>709 , T<733.
:- simpleEvent(T,P,running), T>709 , T<733.

#show simpleEvent/3.