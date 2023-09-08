
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


person(X) :- happensAt(disappear(X),_).
person(X) :- happensAt(appear(X),_).
person(X) :- happensAt(active(X),_).
person(X) :- happensAt(inactive(X),_).
person(X) :- happensAt(walking(X),_).
person(X) :- happensAt(running(X),_).
person(X) :- coords(X,_,_,_).
person(X) :- orientation(X,_,_).


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
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,25,X2).

	orientation(br1gt_2,12,706). coords(br1gt_2,86,209,706).  visible(br1gt_2,706). happensAt(inactive(br1gt_2),706). orientation(br1gt_3,103,706). coords(br1gt_3,299,96,706).  visible(br1gt_3,706). happensAt(active(br1gt_3),706).
orientation(br1gt_2,12,707). coords(br1gt_2,86,209,707).  visible(br1gt_2,707). happensAt(inactive(br1gt_2),707). orientation(br1gt_3,103,707). coords(br1gt_3,299,96,707).  visible(br1gt_3,707). happensAt(active(br1gt_3),707).
orientation(br1gt_2,12,708). coords(br1gt_2,86,209,708).  visible(br1gt_2,708). happensAt(inactive(br1gt_2),708). orientation(br1gt_3,103,708). coords(br1gt_3,299,96,708).  visible(br1gt_3,708). happensAt(active(br1gt_3),708).
orientation(br1gt_2,12,709). coords(br1gt_2,86,209,709).  visible(br1gt_2,709). happensAt(inactive(br1gt_2),709). orientation(br1gt_3,103,709). coords(br1gt_3,299,96,709).  visible(br1gt_3,709). happensAt(active(br1gt_3),709).
orientation(br1gt_2,12,710). coords(br1gt_2,86,209,710).  visible(br1gt_2,710). happensAt(inactive(br1gt_2),710). orientation(br1gt_3,103,710). coords(br1gt_3,299,96,710).  visible(br1gt_3,710). happensAt(active(br1gt_3),710).
orientation(br1gt_2,12,711). coords(br1gt_2,86,209,711).  visible(br1gt_2,711). happensAt(inactive(br1gt_2),711). orientation(br1gt_3,107,711). coords(br1gt_3,299,96,711).  visible(br1gt_3,711). happensAt(active(br1gt_3),711).
orientation(br1gt_2,12,712). coords(br1gt_2,86,209,712).  visible(br1gt_2,712). happensAt(inactive(br1gt_2),712). orientation(br1gt_3,107,712). coords(br1gt_3,299,96,712).  visible(br1gt_3,712). happensAt(active(br1gt_3),712).
orientation(br1gt_2,12,713). coords(br1gt_2,86,209,713).  visible(br1gt_2,713). happensAt(inactive(br1gt_2),713). orientation(br1gt_3,107,713). coords(br1gt_3,298,95,713).  visible(br1gt_3,713). happensAt(active(br1gt_3),713).
orientation(br1gt_2,12,714). coords(br1gt_2,86,209,714).  visible(br1gt_2,714). happensAt(inactive(br1gt_2),714). orientation(br1gt_3,107,714). coords(br1gt_3,298,95,714).  visible(br1gt_3,714). happensAt(active(br1gt_3),714).
orientation(br1gt_2,12,715). coords(br1gt_2,86,209,715).  visible(br1gt_2,715). happensAt(inactive(br1gt_2),715). orientation(br1gt_3,107,715). coords(br1gt_3,298,95,715).  visible(br1gt_3,715). happensAt(active(br1gt_3),715).
orientation(br1gt_2,12,716). coords(br1gt_2,86,209,716).  visible(br1gt_2,716). happensAt(inactive(br1gt_2),716). orientation(br1gt_3,107,716). coords(br1gt_3,298,95,716).  visible(br1gt_3,716). happensAt(active(br1gt_3),716).
orientation(br1gt_2,12,717). coords(br1gt_2,86,209,717).  visible(br1gt_2,717). happensAt(inactive(br1gt_2),717). orientation(br1gt_3,107,717). coords(br1gt_3,298,95,717).  visible(br1gt_3,717). happensAt(active(br1gt_3),717).
orientation(br1gt_2,12,718). coords(br1gt_2,86,209,718).  visible(br1gt_2,718). happensAt(inactive(br1gt_2),718). orientation(br1gt_3,107,718). coords(br1gt_3,298,95,718).  visible(br1gt_3,718). happensAt(active(br1gt_3),718).
orientation(br1gt_2,12,719). coords(br1gt_2,86,209,719).  visible(br1gt_2,719). happensAt(inactive(br1gt_2),719). orientation(br1gt_3,107,719). coords(br1gt_3,298,95,719).  visible(br1gt_3,719). happensAt(active(br1gt_3),719).
orientation(br1gt_2,12,720). coords(br1gt_2,86,209,720).  visible(br1gt_2,720). happensAt(inactive(br1gt_2),720). orientation(br1gt_3,107,720). coords(br1gt_3,298,95,720).  visible(br1gt_3,720). happensAt(active(br1gt_3),720).
orientation(br1gt_2,12,721). coords(br1gt_2,86,209,721).  visible(br1gt_2,721). happensAt(inactive(br1gt_2),721). orientation(br1gt_3,107,721). coords(br1gt_3,298,95,721).  visible(br1gt_3,721). happensAt(active(br1gt_3),721).
orientation(br1gt_2,12,722). coords(br1gt_2,86,209,722).  visible(br1gt_2,722). happensAt(inactive(br1gt_2),722). orientation(br1gt_3,107,722). coords(br1gt_3,298,95,722).  visible(br1gt_3,722). happensAt(active(br1gt_3),722).
orientation(br1gt_2,12,723). coords(br1gt_2,86,209,723).  visible(br1gt_2,723). happensAt(inactive(br1gt_2),723). orientation(br1gt_3,107,723). coords(br1gt_3,299,95,723).  visible(br1gt_3,723). happensAt(active(br1gt_3),723).
orientation(br1gt_2,12,724). coords(br1gt_2,86,209,724).  visible(br1gt_2,724). happensAt(inactive(br1gt_2),724). orientation(br1gt_3,107,724). coords(br1gt_3,298,95,724).  visible(br1gt_3,724). happensAt(active(br1gt_3),724).
orientation(br1gt_2,12,725). coords(br1gt_2,86,209,725).  visible(br1gt_2,725). happensAt(inactive(br1gt_2),725). orientation(br1gt_3,107,725). coords(br1gt_3,298,95,725).  visible(br1gt_3,725). happensAt(active(br1gt_3),725).
orientation(br1gt_2,12,726). coords(br1gt_2,86,209,726).  visible(br1gt_2,726). happensAt(inactive(br1gt_2),726). orientation(br1gt_3,107,726). coords(br1gt_3,298,95,726).  visible(br1gt_3,726). happensAt(active(br1gt_3),726).
orientation(br1gt_2,12,727). coords(br1gt_2,86,209,727).  visible(br1gt_2,727). happensAt(inactive(br1gt_2),727). orientation(br1gt_3,107,727). coords(br1gt_3,298,95,727).  visible(br1gt_3,727). happensAt(active(br1gt_3),727).
orientation(br1gt_2,12,728). coords(br1gt_2,86,209,728).  visible(br1gt_2,728). happensAt(inactive(br1gt_2),728). orientation(br1gt_3,100,728). coords(br1gt_3,298,95,728).  visible(br1gt_3,728). happensAt(active(br1gt_3),728).
orientation(br1gt_2,12,729). coords(br1gt_2,86,209,729).  visible(br1gt_2,729). happensAt(inactive(br1gt_2),729). orientation(br1gt_3,100,729). coords(br1gt_3,298,95,729).  visible(br1gt_3,729). happensAt(active(br1gt_3),729).
orientation(br1gt_2,12,730). coords(br1gt_2,86,209,730).  visible(br1gt_2,730). happensAt(inactive(br1gt_2),730). orientation(br1gt_3,100,730). coords(br1gt_3,298,95,730).  visible(br1gt_3,730). happensAt(active(br1gt_3),730).
#show holdsAt/2.