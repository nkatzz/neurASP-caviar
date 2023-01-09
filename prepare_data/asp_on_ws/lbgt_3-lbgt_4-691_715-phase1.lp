
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

	orientation(lbgt_3,61,691). coords(lbgt_3,146,162,691).  visible(lbgt_3,691). happensAt(inactive(lbgt_3),691). orientation(lbgt_4,158,691). coords(lbgt_4,122,184,691).  visible(lbgt_4,691). happensAt(walking(lbgt_4),691).
orientation(lbgt_3,61,692). coords(lbgt_3,146,162,692).  visible(lbgt_3,692). happensAt(inactive(lbgt_3),692). orientation(lbgt_4,155,692). coords(lbgt_4,122,184,692).  visible(lbgt_4,692). happensAt(walking(lbgt_4),692).
orientation(lbgt_3,61,693). coords(lbgt_3,146,162,693).  visible(lbgt_3,693). happensAt(inactive(lbgt_3),693). orientation(lbgt_4,155,693). coords(lbgt_4,122,183,693).  visible(lbgt_4,693). happensAt(walking(lbgt_4),693).
orientation(lbgt_3,61,694). coords(lbgt_3,146,162,694).  visible(lbgt_3,694). happensAt(inactive(lbgt_3),694). orientation(lbgt_4,153,694). coords(lbgt_4,122,183,694).  visible(lbgt_4,694). happensAt(walking(lbgt_4),694).
orientation(lbgt_3,61,695). coords(lbgt_3,146,162,695).  visible(lbgt_3,695). happensAt(inactive(lbgt_3),695). orientation(lbgt_4,152,695). coords(lbgt_4,121,182,695).  visible(lbgt_4,695). happensAt(walking(lbgt_4),695).
orientation(lbgt_3,61,696). coords(lbgt_3,146,162,696).  visible(lbgt_3,696). happensAt(inactive(lbgt_3),696). orientation(lbgt_4,152,696). coords(lbgt_4,121,182,696).  visible(lbgt_4,696). happensAt(walking(lbgt_4),696).
orientation(lbgt_3,61,697). coords(lbgt_3,146,162,697).  visible(lbgt_3,697). happensAt(inactive(lbgt_3),697). orientation(lbgt_4,152,697). coords(lbgt_4,121,181,697).  visible(lbgt_4,697). happensAt(walking(lbgt_4),697).
orientation(lbgt_3,61,698). coords(lbgt_3,146,162,698).  visible(lbgt_3,698). happensAt(inactive(lbgt_3),698). orientation(lbgt_4,154,698). coords(lbgt_4,121,180,698).  visible(lbgt_4,698). happensAt(walking(lbgt_4),698).
orientation(lbgt_3,61,699). coords(lbgt_3,146,162,699).  visible(lbgt_3,699). happensAt(inactive(lbgt_3),699). orientation(lbgt_4,152,699). coords(lbgt_4,119,179,699).  visible(lbgt_4,699). happensAt(walking(lbgt_4),699).
orientation(lbgt_3,61,700). coords(lbgt_3,146,162,700).  visible(lbgt_3,700). happensAt(inactive(lbgt_3),700). orientation(lbgt_4,152,700). coords(lbgt_4,118,177,700).  visible(lbgt_4,700). happensAt(walking(lbgt_4),700).
orientation(lbgt_3,61,701). coords(lbgt_3,146,162,701).  visible(lbgt_3,701). happensAt(inactive(lbgt_3),701). orientation(lbgt_4,157,701). coords(lbgt_4,117,175,701).  visible(lbgt_4,701). happensAt(walking(lbgt_4),701).
orientation(lbgt_3,61,702). coords(lbgt_3,146,162,702).  visible(lbgt_3,702). happensAt(inactive(lbgt_3),702). orientation(lbgt_4,157,702). coords(lbgt_4,116,173,702).  visible(lbgt_4,702). happensAt(walking(lbgt_4),702).
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
#show holdsAt/2.