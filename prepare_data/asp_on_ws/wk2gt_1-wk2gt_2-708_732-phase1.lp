
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

	orientation(wk2gt_1,125,708). coords(wk2gt_1,114,53,708).  visible(wk2gt_1,708). happensAt(inactive(wk2gt_1),708). orientation(wk2gt_2,142,708). coords(wk2gt_2,99,63,708).  visible(wk2gt_2,708). happensAt(inactive(wk2gt_2),708).
orientation(wk2gt_1,125,709). coords(wk2gt_1,114,53,709).  visible(wk2gt_1,709). happensAt(inactive(wk2gt_1),709). orientation(wk2gt_2,142,709). coords(wk2gt_2,99,63,709).  visible(wk2gt_2,709). happensAt(inactive(wk2gt_2),709).
orientation(wk2gt_1,125,710). coords(wk2gt_1,114,53,710).  visible(wk2gt_1,710). happensAt(inactive(wk2gt_1),710). orientation(wk2gt_2,142,710). coords(wk2gt_2,99,63,710).  visible(wk2gt_2,710). happensAt(inactive(wk2gt_2),710).
orientation(wk2gt_1,125,711). coords(wk2gt_1,114,53,711).  visible(wk2gt_1,711). happensAt(inactive(wk2gt_1),711). orientation(wk2gt_2,142,711). coords(wk2gt_2,99,63,711).  visible(wk2gt_2,711). happensAt(inactive(wk2gt_2),711).
orientation(wk2gt_1,125,712). coords(wk2gt_1,114,53,712).  visible(wk2gt_1,712). happensAt(inactive(wk2gt_1),712). orientation(wk2gt_2,142,712). coords(wk2gt_2,99,63,712).  visible(wk2gt_2,712). happensAt(inactive(wk2gt_2),712).
orientation(wk2gt_1,125,713). coords(wk2gt_1,114,53,713).  visible(wk2gt_1,713). happensAt(inactive(wk2gt_1),713). orientation(wk2gt_2,142,713). coords(wk2gt_2,99,63,713).  visible(wk2gt_2,713). happensAt(inactive(wk2gt_2),713).
orientation(wk2gt_1,125,714). coords(wk2gt_1,114,53,714).  visible(wk2gt_1,714). happensAt(inactive(wk2gt_1),714). orientation(wk2gt_2,142,714). coords(wk2gt_2,99,63,714).  visible(wk2gt_2,714). happensAt(inactive(wk2gt_2),714).
orientation(wk2gt_1,125,715). coords(wk2gt_1,114,53,715).  visible(wk2gt_1,715). happensAt(inactive(wk2gt_1),715). orientation(wk2gt_2,142,715). coords(wk2gt_2,99,63,715).  visible(wk2gt_2,715). happensAt(inactive(wk2gt_2),715).
orientation(wk2gt_1,125,716). coords(wk2gt_1,114,53,716).  visible(wk2gt_1,716). happensAt(inactive(wk2gt_1),716). orientation(wk2gt_2,142,716). coords(wk2gt_2,99,63,716).  visible(wk2gt_2,716). happensAt(inactive(wk2gt_2),716).
orientation(wk2gt_1,125,717). coords(wk2gt_1,114,53,717).  visible(wk2gt_1,717). happensAt(inactive(wk2gt_1),717). orientation(wk2gt_2,142,717). coords(wk2gt_2,99,63,717).  visible(wk2gt_2,717). happensAt(inactive(wk2gt_2),717).
orientation(wk2gt_1,125,718). coords(wk2gt_1,114,53,718).  visible(wk2gt_1,718). happensAt(inactive(wk2gt_1),718). orientation(wk2gt_2,142,718). coords(wk2gt_2,99,63,718).  visible(wk2gt_2,718). happensAt(inactive(wk2gt_2),718).
orientation(wk2gt_1,125,719). coords(wk2gt_1,114,53,719).  visible(wk2gt_1,719). happensAt(inactive(wk2gt_1),719). orientation(wk2gt_2,142,719). coords(wk2gt_2,99,63,719).  visible(wk2gt_2,719). happensAt(inactive(wk2gt_2),719).
orientation(wk2gt_1,125,720). coords(wk2gt_1,114,53,720).  visible(wk2gt_1,720). happensAt(inactive(wk2gt_1),720). orientation(wk2gt_2,142,720). coords(wk2gt_2,99,63,720).  visible(wk2gt_2,720). happensAt(inactive(wk2gt_2),720).
orientation(wk2gt_1,125,721). coords(wk2gt_1,114,53,721).  visible(wk2gt_1,721). happensAt(inactive(wk2gt_1),721). orientation(wk2gt_2,142,721). coords(wk2gt_2,99,63,721).  visible(wk2gt_2,721). happensAt(inactive(wk2gt_2),721).
orientation(wk2gt_1,125,722). coords(wk2gt_1,114,53,722).  visible(wk2gt_1,722). happensAt(inactive(wk2gt_1),722). orientation(wk2gt_2,142,722). coords(wk2gt_2,99,63,722).  visible(wk2gt_2,722). happensAt(inactive(wk2gt_2),722).
orientation(wk2gt_1,125,723). coords(wk2gt_1,114,53,723).  visible(wk2gt_1,723). happensAt(inactive(wk2gt_1),723). orientation(wk2gt_2,142,723). coords(wk2gt_2,99,63,723).  visible(wk2gt_2,723). happensAt(inactive(wk2gt_2),723).
orientation(wk2gt_1,125,724). coords(wk2gt_1,114,53,724).  visible(wk2gt_1,724). happensAt(inactive(wk2gt_1),724). orientation(wk2gt_2,142,724). coords(wk2gt_2,99,63,724).  visible(wk2gt_2,724). happensAt(inactive(wk2gt_2),724).
orientation(wk2gt_1,125,725). coords(wk2gt_1,114,53,725).  visible(wk2gt_1,725). happensAt(inactive(wk2gt_1),725). orientation(wk2gt_2,142,725). coords(wk2gt_2,99,63,725).  visible(wk2gt_2,725). happensAt(inactive(wk2gt_2),725).
orientation(wk2gt_1,125,726). coords(wk2gt_1,114,53,726).  visible(wk2gt_1,726). happensAt(inactive(wk2gt_1),726). orientation(wk2gt_2,142,726). coords(wk2gt_2,99,63,726).  visible(wk2gt_2,726). happensAt(inactive(wk2gt_2),726).
orientation(wk2gt_1,125,727). coords(wk2gt_1,114,53,727).  visible(wk2gt_1,727). happensAt(inactive(wk2gt_1),727). orientation(wk2gt_2,142,727). coords(wk2gt_2,99,63,727).  visible(wk2gt_2,727). happensAt(inactive(wk2gt_2),727).
orientation(wk2gt_1,125,728). coords(wk2gt_1,114,53,728).  visible(wk2gt_1,728). happensAt(inactive(wk2gt_1),728). orientation(wk2gt_2,142,728). coords(wk2gt_2,99,63,728).  visible(wk2gt_2,728). happensAt(inactive(wk2gt_2),728).
orientation(wk2gt_1,125,729). coords(wk2gt_1,114,53,729).  visible(wk2gt_1,729). happensAt(inactive(wk2gt_1),729). orientation(wk2gt_2,142,729). coords(wk2gt_2,99,63,729).  visible(wk2gt_2,729). happensAt(inactive(wk2gt_2),729).
orientation(wk2gt_1,125,730). coords(wk2gt_1,114,53,730).  visible(wk2gt_1,730). happensAt(inactive(wk2gt_1),730). orientation(wk2gt_2,142,730). coords(wk2gt_2,99,63,730).  visible(wk2gt_2,730). happensAt(inactive(wk2gt_2),730).
orientation(wk2gt_1,125,731). coords(wk2gt_1,114,53,731).  visible(wk2gt_1,731). happensAt(inactive(wk2gt_1),731). orientation(wk2gt_2,142,731). coords(wk2gt_2,99,63,731).  visible(wk2gt_2,731). happensAt(inactive(wk2gt_2),731).
orientation(wk2gt_1,125,732). coords(wk2gt_1,114,53,732).  visible(wk2gt_1,732). happensAt(inactive(wk2gt_1),732). orientation(wk2gt_2,142,732). coords(wk2gt_2,99,63,732).  visible(wk2gt_2,732). happensAt(inactive(wk2gt_2),732).
#show holdsAt/2.