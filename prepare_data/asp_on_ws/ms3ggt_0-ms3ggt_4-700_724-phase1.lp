
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

	orientation(ms3ggt_0,4,700). coords(ms3ggt_0,196,220,700).  visible(ms3ggt_0,700). happensAt(walking(ms3ggt_0),700). orientation(ms3ggt_4,129,700). coords(ms3ggt_4,165,18,700).  happensAt(appear(ms3ggt_4),700). happensAt(walking(ms3ggt_4),700).
orientation(ms3ggt_0,4,701). coords(ms3ggt_0,199,220,701).  visible(ms3ggt_0,701). happensAt(walking(ms3ggt_0),701). orientation(ms3ggt_4,129,701). coords(ms3ggt_4,165,18,701).  visible(ms3ggt_4,701). happensAt(walking(ms3ggt_4),701).
orientation(ms3ggt_0,178,702). coords(ms3ggt_0,203,222,702).  visible(ms3ggt_0,702). happensAt(walking(ms3ggt_0),702). orientation(ms3ggt_4,129,702). coords(ms3ggt_4,165,18,702).  visible(ms3ggt_4,702). happensAt(walking(ms3ggt_4),702).
orientation(ms3ggt_0,178,703). coords(ms3ggt_0,205,223,703).  visible(ms3ggt_0,703). happensAt(walking(ms3ggt_0),703). orientation(ms3ggt_4,129,703). coords(ms3ggt_4,165,18,703).  visible(ms3ggt_4,703). happensAt(walking(ms3ggt_4),703).
orientation(ms3ggt_0,175,704). coords(ms3ggt_0,206,223,704).  visible(ms3ggt_0,704). happensAt(walking(ms3ggt_0),704). orientation(ms3ggt_4,129,704). coords(ms3ggt_4,165,18,704).  visible(ms3ggt_4,704). happensAt(walking(ms3ggt_4),704).
orientation(ms3ggt_0,175,705). coords(ms3ggt_0,208,224,705).  visible(ms3ggt_0,705). happensAt(walking(ms3ggt_0),705). orientation(ms3ggt_4,129,705). coords(ms3ggt_4,165,18,705).  visible(ms3ggt_4,705). happensAt(walking(ms3ggt_4),705).
orientation(ms3ggt_0,177,706). coords(ms3ggt_0,210,225,706).  visible(ms3ggt_0,706). happensAt(walking(ms3ggt_0),706). orientation(ms3ggt_4,129,706). coords(ms3ggt_4,165,18,706).  visible(ms3ggt_4,706). happensAt(walking(ms3ggt_4),706).
orientation(ms3ggt_0,178,707). coords(ms3ggt_0,211,226,707).  visible(ms3ggt_0,707). happensAt(walking(ms3ggt_0),707). orientation(ms3ggt_4,129,707). coords(ms3ggt_4,165,18,707).  visible(ms3ggt_4,707). happensAt(walking(ms3ggt_4),707).
orientation(ms3ggt_0,3,708). coords(ms3ggt_0,212,227,708).  visible(ms3ggt_0,708). happensAt(walking(ms3ggt_0),708). orientation(ms3ggt_4,129,708). coords(ms3ggt_4,165,18,708).  visible(ms3ggt_4,708). happensAt(walking(ms3ggt_4),708).
orientation(ms3ggt_0,3,709). coords(ms3ggt_0,213,228,709).  visible(ms3ggt_0,709). happensAt(walking(ms3ggt_0),709). orientation(ms3ggt_4,129,709). coords(ms3ggt_4,165,18,709).  visible(ms3ggt_4,709). happensAt(walking(ms3ggt_4),709).
orientation(ms3ggt_0,3,710). coords(ms3ggt_0,214,230,710).  visible(ms3ggt_0,710). happensAt(walking(ms3ggt_0),710). orientation(ms3ggt_4,129,710). coords(ms3ggt_4,165,18,710).  visible(ms3ggt_4,710). happensAt(walking(ms3ggt_4),710).
orientation(ms3ggt_0,3,711). coords(ms3ggt_0,216,231,711).  visible(ms3ggt_0,711). happensAt(walking(ms3ggt_0),711). orientation(ms3ggt_4,129,711). coords(ms3ggt_4,165,18,711).  visible(ms3ggt_4,711). happensAt(walking(ms3ggt_4),711).
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
#show holdsAt/2.