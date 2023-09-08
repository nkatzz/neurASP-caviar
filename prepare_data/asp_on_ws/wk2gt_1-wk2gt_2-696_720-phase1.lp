
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

	orientation(wk2gt_1,125,696). coords(wk2gt_1,114,53,696).  visible(wk2gt_1,696). happensAt(inactive(wk2gt_1),696). orientation(wk2gt_2,142,696). coords(wk2gt_2,99,63,696).  visible(wk2gt_2,696). happensAt(inactive(wk2gt_2),696).
orientation(wk2gt_1,125,697). coords(wk2gt_1,114,53,697).  visible(wk2gt_1,697). happensAt(inactive(wk2gt_1),697). orientation(wk2gt_2,142,697). coords(wk2gt_2,99,63,697).  visible(wk2gt_2,697). happensAt(inactive(wk2gt_2),697).
orientation(wk2gt_1,125,698). coords(wk2gt_1,114,53,698).  visible(wk2gt_1,698). happensAt(inactive(wk2gt_1),698). orientation(wk2gt_2,142,698). coords(wk2gt_2,99,63,698).  visible(wk2gt_2,698). happensAt(inactive(wk2gt_2),698).
orientation(wk2gt_1,125,699). coords(wk2gt_1,114,53,699).  visible(wk2gt_1,699). happensAt(inactive(wk2gt_1),699). orientation(wk2gt_2,142,699). coords(wk2gt_2,99,63,699).  visible(wk2gt_2,699). happensAt(inactive(wk2gt_2),699).
orientation(wk2gt_1,125,700). coords(wk2gt_1,114,53,700).  visible(wk2gt_1,700). happensAt(inactive(wk2gt_1),700). orientation(wk2gt_2,142,700). coords(wk2gt_2,99,63,700).  visible(wk2gt_2,700). happensAt(inactive(wk2gt_2),700).
orientation(wk2gt_1,125,701). coords(wk2gt_1,114,53,701).  visible(wk2gt_1,701). happensAt(inactive(wk2gt_1),701). orientation(wk2gt_2,142,701). coords(wk2gt_2,99,63,701).  visible(wk2gt_2,701). happensAt(inactive(wk2gt_2),701).
orientation(wk2gt_1,125,702). coords(wk2gt_1,114,53,702).  visible(wk2gt_1,702). happensAt(inactive(wk2gt_1),702). orientation(wk2gt_2,142,702). coords(wk2gt_2,99,63,702).  visible(wk2gt_2,702). happensAt(inactive(wk2gt_2),702).
orientation(wk2gt_1,125,703). coords(wk2gt_1,114,53,703).  visible(wk2gt_1,703). happensAt(inactive(wk2gt_1),703). orientation(wk2gt_2,142,703). coords(wk2gt_2,99,63,703).  visible(wk2gt_2,703). happensAt(inactive(wk2gt_2),703).
orientation(wk2gt_1,125,704). coords(wk2gt_1,114,53,704).  visible(wk2gt_1,704). happensAt(inactive(wk2gt_1),704). orientation(wk2gt_2,142,704). coords(wk2gt_2,99,63,704).  visible(wk2gt_2,704). happensAt(inactive(wk2gt_2),704).
orientation(wk2gt_1,125,705). coords(wk2gt_1,114,53,705).  visible(wk2gt_1,705). happensAt(inactive(wk2gt_1),705). orientation(wk2gt_2,142,705). coords(wk2gt_2,99,63,705).  visible(wk2gt_2,705). happensAt(inactive(wk2gt_2),705).
orientation(wk2gt_1,125,706). coords(wk2gt_1,114,53,706).  visible(wk2gt_1,706). happensAt(inactive(wk2gt_1),706). orientation(wk2gt_2,142,706). coords(wk2gt_2,99,63,706).  visible(wk2gt_2,706). happensAt(inactive(wk2gt_2),706).
orientation(wk2gt_1,125,707). coords(wk2gt_1,114,53,707).  visible(wk2gt_1,707). happensAt(inactive(wk2gt_1),707). orientation(wk2gt_2,142,707). coords(wk2gt_2,99,63,707).  visible(wk2gt_2,707). happensAt(inactive(wk2gt_2),707).
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
#show holdsAt/2.