
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

	orientation(lbgt_3,61,703). coords(lbgt_3,146,162,703).  visible(lbgt_3,703). happensAt(inactive(lbgt_3),703). orientation(lbgt_4,157,703). coords(lbgt_4,116,172,703).  visible(lbgt_4,703). happensAt(walking(lbgt_4),703).
orientation(lbgt_3,61,704). coords(lbgt_3,146,162,704).  visible(lbgt_3,704). happensAt(inactive(lbgt_3),704). orientation(lbgt_4,161,704). coords(lbgt_4,116,171,704).  visible(lbgt_4,704). happensAt(walking(lbgt_4),704).
orientation(lbgt_3,61,705). coords(lbgt_3,146,162,705).  visible(lbgt_3,705). happensAt(inactive(lbgt_3),705). orientation(lbgt_4,161,705). coords(lbgt_4,116,171,705).  visible(lbgt_4,705). happensAt(walking(lbgt_4),705).
orientation(lbgt_3,61,706). coords(lbgt_3,146,162,706).  visible(lbgt_3,706). happensAt(inactive(lbgt_3),706). orientation(lbgt_4,161,706). coords(lbgt_4,115,171,706).  visible(lbgt_4,706). happensAt(walking(lbgt_4),706).
orientation(lbgt_3,61,707). coords(lbgt_3,146,162,707).  visible(lbgt_3,707). happensAt(inactive(lbgt_3),707). orientation(lbgt_4,161,707). coords(lbgt_4,115,171,707).  visible(lbgt_4,707). happensAt(walking(lbgt_4),707).
orientation(lbgt_3,61,708). coords(lbgt_3,146,162,708).  visible(lbgt_3,708). happensAt(inactive(lbgt_3),708). orientation(lbgt_4,161,708). coords(lbgt_4,115,171,708).  visible(lbgt_4,708). happensAt(walking(lbgt_4),708).
orientation(lbgt_3,61,709). coords(lbgt_3,146,162,709).  visible(lbgt_3,709). happensAt(inactive(lbgt_3),709). orientation(lbgt_4,161,709). coords(lbgt_4,114,170,709).  visible(lbgt_4,709). happensAt(walking(lbgt_4),709).
orientation(lbgt_3,61,710). coords(lbgt_3,146,162,710).  visible(lbgt_3,710). happensAt(inactive(lbgt_3),710). orientation(lbgt_4,161,710). coords(lbgt_4,113,169,710).  visible(lbgt_4,710). happensAt(walking(lbgt_4),710).
orientation(lbgt_3,61,711). coords(lbgt_3,146,162,711).  visible(lbgt_3,711). happensAt(inactive(lbgt_3),711). orientation(lbgt_4,161,711). coords(lbgt_4,112,168,711).  visible(lbgt_4,711). happensAt(walking(lbgt_4),711).
orientation(lbgt_3,61,712). coords(lbgt_3,146,162,712).  visible(lbgt_3,712). happensAt(inactive(lbgt_3),712). orientation(lbgt_4,161,712). coords(lbgt_4,111,166,712).  visible(lbgt_4,712). happensAt(walking(lbgt_4),712).
orientation(lbgt_3,61,713). coords(lbgt_3,146,162,713).  visible(lbgt_3,713). happensAt(inactive(lbgt_3),713). orientation(lbgt_4,164,713). coords(lbgt_4,110,166,713).  visible(lbgt_4,713). happensAt(walking(lbgt_4),713).
orientation(lbgt_3,61,714). coords(lbgt_3,146,162,714).  visible(lbgt_3,714). happensAt(inactive(lbgt_3),714). orientation(lbgt_4,164,714). coords(lbgt_4,109,165,714).  visible(lbgt_4,714). happensAt(walking(lbgt_4),714).
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
#show holdsAt/2.