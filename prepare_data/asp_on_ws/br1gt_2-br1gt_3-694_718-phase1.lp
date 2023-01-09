
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

	orientation(br1gt_2,12,694). coords(br1gt_2,86,209,694).  visible(br1gt_2,694). happensAt(inactive(br1gt_2),694). orientation(br1gt_3,96,694). coords(br1gt_3,296,99,694).  visible(br1gt_3,694). happensAt(active(br1gt_3),694).
orientation(br1gt_2,12,695). coords(br1gt_2,86,209,695).  visible(br1gt_2,695). happensAt(inactive(br1gt_2),695). orientation(br1gt_3,96,695). coords(br1gt_3,297,99,695).  visible(br1gt_3,695). happensAt(active(br1gt_3),695).
orientation(br1gt_2,12,696). coords(br1gt_2,86,209,696).  visible(br1gt_2,696). happensAt(inactive(br1gt_2),696). orientation(br1gt_3,96,696). coords(br1gt_3,297,99,696).  visible(br1gt_3,696). happensAt(active(br1gt_3),696).
orientation(br1gt_2,12,697). coords(br1gt_2,86,209,697).  visible(br1gt_2,697). happensAt(inactive(br1gt_2),697). orientation(br1gt_3,96,697). coords(br1gt_3,297,99,697).  visible(br1gt_3,697). happensAt(active(br1gt_3),697).
orientation(br1gt_2,12,698). coords(br1gt_2,86,209,698).  visible(br1gt_2,698). happensAt(inactive(br1gt_2),698). orientation(br1gt_3,96,698). coords(br1gt_3,297,99,698).  visible(br1gt_3,698). happensAt(active(br1gt_3),698).
orientation(br1gt_2,12,699). coords(br1gt_2,86,209,699).  visible(br1gt_2,699). happensAt(inactive(br1gt_2),699). orientation(br1gt_3,96,699). coords(br1gt_3,297,98,699).  visible(br1gt_3,699). happensAt(active(br1gt_3),699).
orientation(br1gt_2,12,700). coords(br1gt_2,86,209,700).  visible(br1gt_2,700). happensAt(inactive(br1gt_2),700). orientation(br1gt_3,96,700). coords(br1gt_3,297,98,700).  visible(br1gt_3,700). happensAt(active(br1gt_3),700).
orientation(br1gt_2,12,701). coords(br1gt_2,86,209,701).  visible(br1gt_2,701). happensAt(inactive(br1gt_2),701). orientation(br1gt_3,96,701). coords(br1gt_3,297,98,701).  visible(br1gt_3,701). happensAt(active(br1gt_3),701).
orientation(br1gt_2,12,702). coords(br1gt_2,86,209,702).  visible(br1gt_2,702). happensAt(inactive(br1gt_2),702). orientation(br1gt_3,96,702). coords(br1gt_3,297,98,702).  visible(br1gt_3,702). happensAt(active(br1gt_3),702).
orientation(br1gt_2,12,703). coords(br1gt_2,86,209,703).  visible(br1gt_2,703). happensAt(inactive(br1gt_2),703). orientation(br1gt_3,103,703). coords(br1gt_3,299,97,703).  visible(br1gt_3,703). happensAt(active(br1gt_3),703).
orientation(br1gt_2,12,704). coords(br1gt_2,86,209,704).  visible(br1gt_2,704). happensAt(inactive(br1gt_2),704). orientation(br1gt_3,103,704). coords(br1gt_3,299,97,704).  visible(br1gt_3,704). happensAt(active(br1gt_3),704).
orientation(br1gt_2,12,705). coords(br1gt_2,86,209,705).  visible(br1gt_2,705). happensAt(inactive(br1gt_2),705). orientation(br1gt_3,103,705). coords(br1gt_3,299,96,705).  visible(br1gt_3,705). happensAt(active(br1gt_3),705).
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
#show holdsAt/2.