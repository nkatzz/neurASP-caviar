
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

	orientation(lbpugt_2,126,707). coords(lbpugt_2,265,88,707).  visible(lbpugt_2,707). happensAt(walking(lbpugt_2),707). orientation(lbpugt_4,135,707). coords(lbpugt_4,219,34,707).  visible(lbpugt_4,707). happensAt(inactive(lbpugt_4),707).
orientation(lbpugt_2,126,708). coords(lbpugt_2,265,88,708).  visible(lbpugt_2,708). happensAt(walking(lbpugt_2),708). orientation(lbpugt_4,135,708). coords(lbpugt_4,219,34,708).  visible(lbpugt_4,708). happensAt(inactive(lbpugt_4),708).
orientation(lbpugt_2,126,709). coords(lbpugt_2,266,88,709).  visible(lbpugt_2,709). happensAt(walking(lbpugt_2),709). orientation(lbpugt_4,135,709). coords(lbpugt_4,219,34,709).  visible(lbpugt_4,709). happensAt(inactive(lbpugt_4),709).
orientation(lbpugt_2,126,710). coords(lbpugt_2,267,88,710).  visible(lbpugt_2,710). happensAt(walking(lbpugt_2),710). orientation(lbpugt_4,135,710). coords(lbpugt_4,219,34,710).  visible(lbpugt_4,710). happensAt(inactive(lbpugt_4),710).
orientation(lbpugt_2,126,711). coords(lbpugt_2,267,88,711).  visible(lbpugt_2,711). happensAt(walking(lbpugt_2),711). orientation(lbpugt_4,135,711). coords(lbpugt_4,219,34,711).  visible(lbpugt_4,711). happensAt(inactive(lbpugt_4),711).
orientation(lbpugt_2,118,712). coords(lbpugt_2,268,88,712).  visible(lbpugt_2,712). happensAt(walking(lbpugt_2),712). orientation(lbpugt_4,135,712). coords(lbpugt_4,219,34,712).  visible(lbpugt_4,712). happensAt(inactive(lbpugt_4),712).
orientation(lbpugt_2,121,713). coords(lbpugt_2,272,89,713).  visible(lbpugt_2,713). happensAt(walking(lbpugt_2),713). orientation(lbpugt_4,135,713). coords(lbpugt_4,219,34,713).  visible(lbpugt_4,713). happensAt(inactive(lbpugt_4),713).
orientation(lbpugt_2,121,714). coords(lbpugt_2,276,90,714).  visible(lbpugt_2,714). happensAt(walking(lbpugt_2),714). orientation(lbpugt_4,135,714). coords(lbpugt_4,219,34,714).  visible(lbpugt_4,714). happensAt(inactive(lbpugt_4),714).
orientation(lbpugt_2,127,715). coords(lbpugt_2,279,90,715).  visible(lbpugt_2,715). happensAt(walking(lbpugt_2),715). orientation(lbpugt_4,135,715). coords(lbpugt_4,219,34,715).  visible(lbpugt_4,715). happensAt(inactive(lbpugt_4),715).
orientation(lbpugt_2,127,716). coords(lbpugt_2,282,90,716).  visible(lbpugt_2,716). happensAt(walking(lbpugt_2),716). orientation(lbpugt_4,135,716). coords(lbpugt_4,219,34,716).  visible(lbpugt_4,716). happensAt(inactive(lbpugt_4),716).
orientation(lbpugt_2,127,717). coords(lbpugt_2,283,90,717).  visible(lbpugt_2,717). happensAt(walking(lbpugt_2),717). orientation(lbpugt_4,135,717). coords(lbpugt_4,219,34,717).  visible(lbpugt_4,717). happensAt(inactive(lbpugt_4),717).
orientation(lbpugt_2,127,718). coords(lbpugt_2,285,91,718).  visible(lbpugt_2,718). happensAt(walking(lbpugt_2),718). orientation(lbpugt_4,135,718). coords(lbpugt_4,219,34,718).  visible(lbpugt_4,718). happensAt(inactive(lbpugt_4),718).
orientation(lbpugt_2,127,719). coords(lbpugt_2,286,91,719).  visible(lbpugt_2,719). happensAt(walking(lbpugt_2),719). orientation(lbpugt_4,135,719). coords(lbpugt_4,219,34,719).  visible(lbpugt_4,719). happensAt(inactive(lbpugt_4),719).
orientation(lbpugt_2,127,720). coords(lbpugt_2,287,91,720).  visible(lbpugt_2,720). happensAt(walking(lbpugt_2),720). orientation(lbpugt_4,135,720). coords(lbpugt_4,219,34,720).  visible(lbpugt_4,720). happensAt(inactive(lbpugt_4),720).
orientation(lbpugt_2,127,721). coords(lbpugt_2,288,91,721).  visible(lbpugt_2,721). happensAt(walking(lbpugt_2),721). orientation(lbpugt_4,135,721). coords(lbpugt_4,219,34,721).  visible(lbpugt_4,721). happensAt(inactive(lbpugt_4),721).
orientation(lbpugt_2,124,722). coords(lbpugt_2,289,91,722).  visible(lbpugt_2,722). happensAt(walking(lbpugt_2),722). orientation(lbpugt_4,135,722). coords(lbpugt_4,219,34,722).  visible(lbpugt_4,722). happensAt(inactive(lbpugt_4),722).
orientation(lbpugt_2,121,723). coords(lbpugt_2,289,91,723).  visible(lbpugt_2,723). happensAt(walking(lbpugt_2),723). orientation(lbpugt_4,135,723). coords(lbpugt_4,219,34,723).  visible(lbpugt_4,723). happensAt(inactive(lbpugt_4),723).
orientation(lbpugt_2,121,724). coords(lbpugt_2,290,91,724).  visible(lbpugt_2,724). happensAt(walking(lbpugt_2),724). orientation(lbpugt_4,135,724). coords(lbpugt_4,219,34,724).  visible(lbpugt_4,724). happensAt(inactive(lbpugt_4),724).
orientation(lbpugt_2,116,725). coords(lbpugt_2,290,91,725).  visible(lbpugt_2,725). happensAt(walking(lbpugt_2),725). orientation(lbpugt_4,135,725). coords(lbpugt_4,219,34,725).  visible(lbpugt_4,725). happensAt(inactive(lbpugt_4),725).
orientation(lbpugt_2,113,726). coords(lbpugt_2,290,92,726).  visible(lbpugt_2,726). happensAt(walking(lbpugt_2),726). orientation(lbpugt_4,135,726). coords(lbpugt_4,219,34,726).  visible(lbpugt_4,726). happensAt(inactive(lbpugt_4),726).
orientation(lbpugt_2,108,727). coords(lbpugt_2,291,93,727).  visible(lbpugt_2,727). happensAt(walking(lbpugt_2),727). orientation(lbpugt_4,135,727). coords(lbpugt_4,219,34,727).  visible(lbpugt_4,727). happensAt(inactive(lbpugt_4),727).
orientation(lbpugt_2,102,728). coords(lbpugt_2,291,94,728).  visible(lbpugt_2,728). happensAt(walking(lbpugt_2),728). orientation(lbpugt_4,135,728). coords(lbpugt_4,219,34,728).  visible(lbpugt_4,728). happensAt(inactive(lbpugt_4),728).
orientation(lbpugt_2,104,729). coords(lbpugt_2,293,96,729).  visible(lbpugt_2,729). happensAt(walking(lbpugt_2),729). orientation(lbpugt_4,135,729). coords(lbpugt_4,219,34,729).  visible(lbpugt_4,729). happensAt(inactive(lbpugt_4),729).
orientation(lbpugt_2,102,730). coords(lbpugt_2,293,96,730).  visible(lbpugt_2,730). happensAt(walking(lbpugt_2),730). orientation(lbpugt_4,135,730). coords(lbpugt_4,219,34,730).  visible(lbpugt_4,730). happensAt(inactive(lbpugt_4),730).
orientation(lbpugt_2,102,731). coords(lbpugt_2,293,97,731).  visible(lbpugt_2,731). happensAt(walking(lbpugt_2),731). orientation(lbpugt_4,135,731). coords(lbpugt_4,219,34,731).  visible(lbpugt_4,731). happensAt(inactive(lbpugt_4),731).
#show holdsAt/2.