
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

	orientation(lbgt_3,61,715). coords(lbgt_3,146,162,715).  visible(lbgt_3,715). happensAt(inactive(lbgt_3),715). orientation(lbgt_4,164,715). coords(lbgt_4,108,164,715).  visible(lbgt_4,715). happensAt(walking(lbgt_4),715).
orientation(lbgt_3,61,716). coords(lbgt_3,146,162,716).  visible(lbgt_3,716). happensAt(inactive(lbgt_3),716). orientation(lbgt_4,164,716). coords(lbgt_4,107,163,716).  visible(lbgt_4,716). happensAt(walking(lbgt_4),716).
orientation(lbgt_3,61,717). coords(lbgt_3,146,162,717).  visible(lbgt_3,717). happensAt(inactive(lbgt_3),717). orientation(lbgt_4,164,717). coords(lbgt_4,107,163,717).  visible(lbgt_4,717). happensAt(walking(lbgt_4),717).
orientation(lbgt_3,61,718). coords(lbgt_3,146,162,718).  visible(lbgt_3,718). happensAt(inactive(lbgt_3),718). orientation(lbgt_4,165,718). coords(lbgt_4,107,163,718).  visible(lbgt_4,718). happensAt(walking(lbgt_4),718).
orientation(lbgt_3,61,719). coords(lbgt_3,146,162,719).  visible(lbgt_3,719). happensAt(inactive(lbgt_3),719). orientation(lbgt_4,161,719). coords(lbgt_4,107,162,719).  visible(lbgt_4,719). happensAt(walking(lbgt_4),719).
orientation(lbgt_3,61,720). coords(lbgt_3,146,162,720).  visible(lbgt_3,720). happensAt(inactive(lbgt_3),720). orientation(lbgt_4,158,720). coords(lbgt_4,107,162,720).  visible(lbgt_4,720). happensAt(walking(lbgt_4),720).
orientation(lbgt_3,61,721). coords(lbgt_3,146,162,721).  visible(lbgt_3,721). happensAt(inactive(lbgt_3),721). orientation(lbgt_4,154,721). coords(lbgt_4,107,162,721).  visible(lbgt_4,721). happensAt(walking(lbgt_4),721).
orientation(lbgt_3,61,722). coords(lbgt_3,146,162,722).  visible(lbgt_3,722). happensAt(inactive(lbgt_3),722). orientation(lbgt_4,154,722). coords(lbgt_4,105,161,722).  visible(lbgt_4,722). happensAt(walking(lbgt_4),722).
orientation(lbgt_3,61,723). coords(lbgt_3,146,162,723).  visible(lbgt_3,723). happensAt(inactive(lbgt_3),723). orientation(lbgt_4,154,723). coords(lbgt_4,105,159,723).  visible(lbgt_4,723). happensAt(walking(lbgt_4),723).
orientation(lbgt_3,61,724). coords(lbgt_3,146,162,724).  visible(lbgt_3,724). happensAt(inactive(lbgt_3),724). orientation(lbgt_4,154,724). coords(lbgt_4,104,158,724).  visible(lbgt_4,724). happensAt(walking(lbgt_4),724).
orientation(lbgt_3,61,725). coords(lbgt_3,146,162,725).  visible(lbgt_3,725). happensAt(inactive(lbgt_3),725). orientation(lbgt_4,154,725). coords(lbgt_4,102,156,725).  visible(lbgt_4,725). happensAt(walking(lbgt_4),725).
orientation(lbgt_3,61,726). coords(lbgt_3,146,162,726).  visible(lbgt_3,726). happensAt(inactive(lbgt_3),726). orientation(lbgt_4,154,726). coords(lbgt_4,101,155,726).  visible(lbgt_4,726). happensAt(walking(lbgt_4),726).
orientation(lbgt_3,61,727). coords(lbgt_3,146,162,727).  visible(lbgt_3,727). happensAt(inactive(lbgt_3),727). orientation(lbgt_4,158,727). coords(lbgt_4,101,154,727).  visible(lbgt_4,727). happensAt(walking(lbgt_4),727).
orientation(lbgt_3,61,728). coords(lbgt_3,146,162,728).  visible(lbgt_3,728). happensAt(inactive(lbgt_3),728). orientation(lbgt_4,158,728). coords(lbgt_4,101,152,728).  visible(lbgt_4,728). happensAt(walking(lbgt_4),728).
orientation(lbgt_3,61,729). coords(lbgt_3,146,162,729).  visible(lbgt_3,729). happensAt(inactive(lbgt_3),729). orientation(lbgt_4,158,729). coords(lbgt_4,100,152,729).  visible(lbgt_4,729). happensAt(walking(lbgt_4),729).
orientation(lbgt_3,61,730). coords(lbgt_3,146,162,730).  visible(lbgt_3,730). happensAt(inactive(lbgt_3),730). orientation(lbgt_4,158,730). coords(lbgt_4,100,151,730).  visible(lbgt_4,730). happensAt(walking(lbgt_4),730).
orientation(lbgt_3,61,731). coords(lbgt_3,146,162,731).  visible(lbgt_3,731). happensAt(inactive(lbgt_3),731). orientation(lbgt_4,158,731). coords(lbgt_4,100,151,731).  visible(lbgt_4,731). happensAt(walking(lbgt_4),731).
orientation(lbgt_3,61,732). coords(lbgt_3,146,162,732).  visible(lbgt_3,732). happensAt(inactive(lbgt_3),732). orientation(lbgt_4,153,732). coords(lbgt_4,100,151,732).  visible(lbgt_4,732). happensAt(walking(lbgt_4),732).
orientation(lbgt_3,61,733). coords(lbgt_3,146,162,733).  visible(lbgt_3,733). happensAt(inactive(lbgt_3),733). orientation(lbgt_4,158,733). coords(lbgt_4,100,151,733).  visible(lbgt_4,733). happensAt(walking(lbgt_4),733).
orientation(lbgt_3,61,734). coords(lbgt_3,146,162,734).  visible(lbgt_3,734). happensAt(inactive(lbgt_3),734). orientation(lbgt_4,159,734). coords(lbgt_4,100,150,734).  visible(lbgt_4,734). happensAt(walking(lbgt_4),734).
orientation(lbgt_3,61,735). coords(lbgt_3,146,162,735).  visible(lbgt_3,735). happensAt(inactive(lbgt_3),735). orientation(lbgt_4,159,735). coords(lbgt_4,99,148,735).  visible(lbgt_4,735). happensAt(walking(lbgt_4),735).
orientation(lbgt_3,61,736). coords(lbgt_3,146,162,736).  visible(lbgt_3,736). happensAt(inactive(lbgt_3),736). orientation(lbgt_4,160,736). coords(lbgt_4,98,147,736).  visible(lbgt_4,736). happensAt(walking(lbgt_4),736).
orientation(lbgt_3,61,737). coords(lbgt_3,146,162,737).  visible(lbgt_3,737). happensAt(inactive(lbgt_3),737). orientation(lbgt_4,159,737). coords(lbgt_4,96,146,737).  visible(lbgt_4,737). happensAt(walking(lbgt_4),737).
orientation(lbgt_3,61,738). coords(lbgt_3,146,162,738).  visible(lbgt_3,738). happensAt(inactive(lbgt_3),738). orientation(lbgt_4,159,738). coords(lbgt_4,95,144,738).  visible(lbgt_4,738). happensAt(walking(lbgt_4),738).
orientation(lbgt_3,61,739). coords(lbgt_3,146,162,739).  visible(lbgt_3,739). happensAt(inactive(lbgt_3),739). orientation(lbgt_4,159,739). coords(lbgt_4,95,144,739).  visible(lbgt_4,739). happensAt(walking(lbgt_4),739).
#show holdsAt/2.