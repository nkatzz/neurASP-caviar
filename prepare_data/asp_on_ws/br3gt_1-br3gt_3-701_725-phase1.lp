
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

	orientation(br3gt_1,135,701). coords(br3gt_1,197,60,701).  visible(br3gt_1,701). happensAt(walking(br3gt_1),701). orientation(br3gt_3,147,701). coords(br3gt_3,10,62,701).  visible(br3gt_3,701). happensAt(walking(br3gt_3),701).
orientation(br3gt_1,135,702). coords(br3gt_1,197,61,702).  visible(br3gt_1,702). happensAt(walking(br3gt_1),702). orientation(br3gt_3,147,702). coords(br3gt_3,11,61,702).  visible(br3gt_3,702). happensAt(walking(br3gt_3),702).
orientation(br3gt_1,131,703). coords(br3gt_1,197,62,703).  visible(br3gt_1,703). happensAt(walking(br3gt_1),703). orientation(br3gt_3,147,703). coords(br3gt_3,11,60,703).  visible(br3gt_3,703). happensAt(walking(br3gt_3),703).
orientation(br3gt_1,131,704). coords(br3gt_1,197,62,704).  visible(br3gt_1,704). happensAt(walking(br3gt_1),704). orientation(br3gt_3,147,704). coords(br3gt_3,11,59,704).  visible(br3gt_3,704). happensAt(walking(br3gt_3),704).
orientation(br3gt_1,134,705). coords(br3gt_1,197,62,705).  visible(br3gt_1,705). happensAt(walking(br3gt_1),705). orientation(br3gt_3,147,705). coords(br3gt_3,12,59,705).  visible(br3gt_3,705). happensAt(walking(br3gt_3),705).
orientation(br3gt_1,134,706). coords(br3gt_1,197,63,706).  visible(br3gt_1,706). happensAt(walking(br3gt_1),706). orientation(br3gt_3,147,706). coords(br3gt_3,12,58,706).  visible(br3gt_3,706). happensAt(walking(br3gt_3),706).
orientation(br3gt_1,134,707). coords(br3gt_1,197,63,707).  visible(br3gt_1,707). happensAt(walking(br3gt_1),707). orientation(br3gt_3,147,707). coords(br3gt_3,12,57,707).  visible(br3gt_3,707). happensAt(walking(br3gt_3),707).
orientation(br3gt_1,134,708). coords(br3gt_1,198,64,708).  visible(br3gt_1,708). happensAt(walking(br3gt_1),708). orientation(br3gt_3,145,708). coords(br3gt_3,12,55,708).  visible(br3gt_3,708). happensAt(walking(br3gt_3),708).
orientation(br3gt_1,134,709). coords(br3gt_1,200,64,709).  visible(br3gt_1,709). happensAt(walking(br3gt_1),709). orientation(br3gt_3,145,709). coords(br3gt_3,13,55,709).  visible(br3gt_3,709). happensAt(walking(br3gt_3),709).
orientation(br3gt_1,134,710). coords(br3gt_1,201,65,710).  visible(br3gt_1,710). happensAt(walking(br3gt_1),710). orientation(br3gt_3,145,710). coords(br3gt_3,13,55,710).  visible(br3gt_3,710). happensAt(walking(br3gt_3),710).
orientation(br3gt_1,134,711). coords(br3gt_1,202,68,711).  visible(br3gt_1,711). happensAt(walking(br3gt_1),711). orientation(br3gt_3,145,711). coords(br3gt_3,13,54,711).  visible(br3gt_3,711). happensAt(walking(br3gt_3),711).
orientation(br3gt_1,134,712). coords(br3gt_1,202,70,712).  visible(br3gt_1,712). happensAt(walking(br3gt_1),712). orientation(br3gt_3,145,712). coords(br3gt_3,12,54,712).  visible(br3gt_3,712). happensAt(walking(br3gt_3),712).
orientation(br3gt_1,134,713). coords(br3gt_1,203,71,713).  visible(br3gt_1,713). happensAt(walking(br3gt_1),713). orientation(br3gt_3,145,713). coords(br3gt_3,12,53,713).  visible(br3gt_3,713). happensAt(walking(br3gt_3),713).
orientation(br3gt_1,134,714). coords(br3gt_1,203,72,714).  visible(br3gt_1,714). happensAt(walking(br3gt_1),714). orientation(br3gt_3,145,714). coords(br3gt_3,11,53,714).  visible(br3gt_3,714). happensAt(walking(br3gt_3),714).
orientation(br3gt_1,132,715). coords(br3gt_1,204,73,715).  visible(br3gt_1,715). happensAt(walking(br3gt_1),715). orientation(br3gt_3,145,715). coords(br3gt_3,11,53,715).  visible(br3gt_3,715). happensAt(walking(br3gt_3),715).
orientation(br3gt_1,132,716). coords(br3gt_1,204,74,716).  visible(br3gt_1,716). happensAt(walking(br3gt_1),716). orientation(br3gt_3,145,716). coords(br3gt_3,11,53,716).  happensAt(occluded(br3gt_3),716). happensAt(walking(br3gt_3),716).
orientation(br3gt_1,132,717). coords(br3gt_1,204,74,717).  visible(br3gt_1,717). happensAt(walking(br3gt_1),717). orientation(br3gt_3,145,717). coords(br3gt_3,11,53,717).  happensAt(occluded(br3gt_3),717). happensAt(walking(br3gt_3),717).
orientation(br3gt_1,132,718). coords(br3gt_1,205,75,718).  visible(br3gt_1,718). happensAt(walking(br3gt_1),718). orientation(br3gt_3,145,718). coords(br3gt_3,11,53,718).  happensAt(occluded(br3gt_3),718). happensAt(walking(br3gt_3),718).
orientation(br3gt_1,132,719). coords(br3gt_1,205,75,719).  visible(br3gt_1,719). happensAt(walking(br3gt_1),719). orientation(br3gt_3,145,719). coords(br3gt_3,11,53,719).  happensAt(occluded(br3gt_3),719). happensAt(walking(br3gt_3),719).
orientation(br3gt_1,132,720). coords(br3gt_1,205,76,720).  visible(br3gt_1,720). happensAt(walking(br3gt_1),720). orientation(br3gt_3,145,720). coords(br3gt_3,11,53,720).  happensAt(occluded(br3gt_3),720). happensAt(walking(br3gt_3),720).
orientation(br3gt_1,132,721). coords(br3gt_1,205,76,721).  visible(br3gt_1,721). happensAt(walking(br3gt_1),721). orientation(br3gt_3,150,721). coords(br3gt_3,16,46,721).  visible(br3gt_3,721). happensAt(walking(br3gt_3),721).
orientation(br3gt_1,132,722). coords(br3gt_1,205,77,722).  visible(br3gt_1,722). happensAt(walking(br3gt_1),722). orientation(br3gt_3,150,722). coords(br3gt_3,16,46,722).  visible(br3gt_3,722). happensAt(walking(br3gt_3),722).
orientation(br3gt_1,130,723). coords(br3gt_1,205,80,723).  visible(br3gt_1,723). happensAt(walking(br3gt_1),723). orientation(br3gt_3,150,723). coords(br3gt_3,16,45,723).  visible(br3gt_3,723). happensAt(walking(br3gt_3),723).
orientation(br3gt_1,130,724). coords(br3gt_1,206,83,724).  visible(br3gt_1,724). happensAt(walking(br3gt_1),724). orientation(br3gt_3,150,724). coords(br3gt_3,16,45,724).  visible(br3gt_3,724). happensAt(walking(br3gt_3),724).
orientation(br3gt_1,130,725). coords(br3gt_1,207,85,725).  visible(br3gt_1,725). happensAt(walking(br3gt_1),725). orientation(br3gt_3,150,725). coords(br3gt_3,16,45,725).  happensAt(disappear(br3gt_3),725). happensAt(walking(br3gt_3),725).
#show holdsAt/2.