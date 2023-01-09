
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

	orientation(ms3ggt_0,3,712). coords(ms3ggt_0,216,233,712).  visible(ms3ggt_0,712). happensAt(walking(ms3ggt_0),712). orientation(ms3ggt_4,129,712). coords(ms3ggt_4,165,18,712).  visible(ms3ggt_4,712). happensAt(walking(ms3ggt_4),712).
orientation(ms3ggt_0,3,713). coords(ms3ggt_0,220,233,713).  visible(ms3ggt_0,713). happensAt(walking(ms3ggt_0),713). orientation(ms3ggt_4,129,713). coords(ms3ggt_4,165,18,713).  visible(ms3ggt_4,713). happensAt(walking(ms3ggt_4),713).
orientation(ms3ggt_0,3,714). coords(ms3ggt_0,222,235,714).  visible(ms3ggt_0,714). happensAt(walking(ms3ggt_0),714). orientation(ms3ggt_4,129,714). coords(ms3ggt_4,165,18,714).  visible(ms3ggt_4,714). happensAt(walking(ms3ggt_4),714).
orientation(ms3ggt_0,5,715). coords(ms3ggt_0,225,237,715).  visible(ms3ggt_0,715). happensAt(walking(ms3ggt_0),715). orientation(ms3ggt_4,129,715). coords(ms3ggt_4,165,18,715).  visible(ms3ggt_4,715). happensAt(walking(ms3ggt_4),715).
orientation(ms3ggt_0,5,716). coords(ms3ggt_0,227,237,716).  visible(ms3ggt_0,716). happensAt(walking(ms3ggt_0),716). orientation(ms3ggt_4,129,716). coords(ms3ggt_4,165,18,716).  visible(ms3ggt_4,716). happensAt(walking(ms3ggt_4),716).
orientation(ms3ggt_0,5,717). coords(ms3ggt_0,229,237,717).  visible(ms3ggt_0,717). happensAt(walking(ms3ggt_0),717). orientation(ms3ggt_4,129,717). coords(ms3ggt_4,165,18,717).  visible(ms3ggt_4,717). happensAt(walking(ms3ggt_4),717).
orientation(ms3ggt_0,13,718). coords(ms3ggt_0,230,238,718).  visible(ms3ggt_0,718). happensAt(walking(ms3ggt_0),718). orientation(ms3ggt_4,129,718). coords(ms3ggt_4,161,19,718).  visible(ms3ggt_4,718). happensAt(walking(ms3ggt_4),718).
orientation(ms3ggt_0,13,719). coords(ms3ggt_0,232,239,719).  visible(ms3ggt_0,719). happensAt(walking(ms3ggt_0),719). orientation(ms3ggt_4,129,719). coords(ms3ggt_4,161,19,719).  visible(ms3ggt_4,719). happensAt(walking(ms3ggt_4),719).
orientation(ms3ggt_0,21,720). coords(ms3ggt_0,232,241,720).  visible(ms3ggt_0,720). happensAt(walking(ms3ggt_0),720). orientation(ms3ggt_4,129,720). coords(ms3ggt_4,161,19,720).  visible(ms3ggt_4,720). happensAt(walking(ms3ggt_4),720).
orientation(ms3ggt_0,21,721). coords(ms3ggt_0,233,243,721).  visible(ms3ggt_0,721). happensAt(walking(ms3ggt_0),721). orientation(ms3ggt_4,129,721). coords(ms3ggt_4,161,19,721).  visible(ms3ggt_4,721). happensAt(walking(ms3ggt_4),721).
orientation(ms3ggt_0,25,722). coords(ms3ggt_0,234,245,722).  visible(ms3ggt_0,722). happensAt(walking(ms3ggt_0),722). orientation(ms3ggt_4,129,722). coords(ms3ggt_4,160,20,722).  visible(ms3ggt_4,722). happensAt(walking(ms3ggt_4),722).
orientation(ms3ggt_0,25,723). coords(ms3ggt_0,236,247,723).  visible(ms3ggt_0,723). happensAt(walking(ms3ggt_0),723). orientation(ms3ggt_4,129,723). coords(ms3ggt_4,159,20,723).  visible(ms3ggt_4,723). happensAt(walking(ms3ggt_4),723).
orientation(ms3ggt_0,25,724). coords(ms3ggt_0,236,249,724).  visible(ms3ggt_0,724). happensAt(walking(ms3ggt_0),724). orientation(ms3ggt_4,129,724). coords(ms3ggt_4,159,20,724).  visible(ms3ggt_4,724). happensAt(walking(ms3ggt_4),724).
orientation(ms3ggt_0,25,725). coords(ms3ggt_0,237,252,725).  visible(ms3ggt_0,725). happensAt(walking(ms3ggt_0),725). orientation(ms3ggt_4,129,725). coords(ms3ggt_4,159,20,725).  visible(ms3ggt_4,725). happensAt(walking(ms3ggt_4),725).
orientation(ms3ggt_0,26,726). coords(ms3ggt_0,240,254,726).  visible(ms3ggt_0,726). happensAt(walking(ms3ggt_0),726). orientation(ms3ggt_4,129,726). coords(ms3ggt_4,159,20,726).  visible(ms3ggt_4,726). happensAt(walking(ms3ggt_4),726).
orientation(ms3ggt_0,26,727). coords(ms3ggt_0,244,256,727).  visible(ms3ggt_0,727). happensAt(walking(ms3ggt_0),727). orientation(ms3ggt_4,129,727). coords(ms3ggt_4,158,20,727).  visible(ms3ggt_4,727). happensAt(walking(ms3ggt_4),727).
orientation(ms3ggt_0,21,728). coords(ms3ggt_0,249,256,728).  visible(ms3ggt_0,728). happensAt(walking(ms3ggt_0),728). orientation(ms3ggt_4,129,728). coords(ms3ggt_4,158,20,728).  visible(ms3ggt_4,728). happensAt(walking(ms3ggt_4),728).
orientation(ms3ggt_0,21,729). coords(ms3ggt_0,250,256,729).  visible(ms3ggt_0,729). happensAt(walking(ms3ggt_0),729). orientation(ms3ggt_4,129,729). coords(ms3ggt_4,158,20,729).  visible(ms3ggt_4,729). happensAt(walking(ms3ggt_4),729).
orientation(ms3ggt_0,21,730). coords(ms3ggt_0,253,257,730).  visible(ms3ggt_0,730). happensAt(walking(ms3ggt_0),730). orientation(ms3ggt_4,129,730). coords(ms3ggt_4,158,20,730).  visible(ms3ggt_4,730). happensAt(walking(ms3ggt_4),730).
orientation(ms3ggt_0,21,731). coords(ms3ggt_0,254,258,731).  visible(ms3ggt_0,731). happensAt(walking(ms3ggt_0),731). orientation(ms3ggt_4,129,731). coords(ms3ggt_4,158,20,731).  visible(ms3ggt_4,731). happensAt(walking(ms3ggt_4),731).
orientation(ms3ggt_0,28,732). coords(ms3ggt_0,255,259,732).  visible(ms3ggt_0,732). happensAt(walking(ms3ggt_0),732). orientation(ms3ggt_4,129,732). coords(ms3ggt_4,157,20,732).  visible(ms3ggt_4,732). happensAt(walking(ms3ggt_4),732).
orientation(ms3ggt_0,28,733). coords(ms3ggt_0,257,260,733).  visible(ms3ggt_0,733). happensAt(walking(ms3ggt_0),733). orientation(ms3ggt_4,129,733). coords(ms3ggt_4,157,20,733).  visible(ms3ggt_4,733). happensAt(walking(ms3ggt_4),733).
orientation(ms3ggt_0,28,734). coords(ms3ggt_0,257,262,734).  visible(ms3ggt_0,734). happensAt(walking(ms3ggt_0),734). orientation(ms3ggt_4,129,734). coords(ms3ggt_4,157,20,734).  visible(ms3ggt_4,734). happensAt(walking(ms3ggt_4),734).
orientation(ms3ggt_0,28,735). coords(ms3ggt_0,258,264,735).  visible(ms3ggt_0,735). happensAt(walking(ms3ggt_0),735). orientation(ms3ggt_4,129,735). coords(ms3ggt_4,157,20,735).  visible(ms3ggt_4,735). happensAt(walking(ms3ggt_4),735).
orientation(ms3ggt_0,28,736). coords(ms3ggt_0,259,266,736).  visible(ms3ggt_0,736). happensAt(walking(ms3ggt_0),736). orientation(ms3ggt_4,129,736). coords(ms3ggt_4,157,20,736).  visible(ms3ggt_4,736). happensAt(walking(ms3ggt_4),736).
#show holdsAt/2.