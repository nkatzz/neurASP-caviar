
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

	orientation(lb1gt_1,162,432). coords(lb1gt_1,165,183,432).  visible(lb1gt_1,432). happensAt(walking(lb1gt_1),432). orientation(lb1gt_2,144,432). coords(lb1gt_2,189,141,432).  visible(lb1gt_2,432). happensAt(active(lb1gt_2),432).
orientation(lb1gt_1,164,433). coords(lb1gt_1,167,183,433).  visible(lb1gt_1,433). happensAt(walking(lb1gt_1),433). orientation(lb1gt_2,144,433). coords(lb1gt_2,189,141,433).  visible(lb1gt_2,433). happensAt(active(lb1gt_2),433).
orientation(lb1gt_1,171,434). coords(lb1gt_1,167,184,434).  visible(lb1gt_1,434). happensAt(walking(lb1gt_1),434). orientation(lb1gt_2,144,434). coords(lb1gt_2,189,142,434).  visible(lb1gt_2,434). happensAt(active(lb1gt_2),434).
orientation(lb1gt_1,174,435). coords(lb1gt_1,168,185,435).  visible(lb1gt_1,435). happensAt(walking(lb1gt_1),435). orientation(lb1gt_2,144,435). coords(lb1gt_2,189,142,435).  visible(lb1gt_2,435). happensAt(active(lb1gt_2),435).
orientation(lb1gt_1,173,436). coords(lb1gt_1,169,186,436).  visible(lb1gt_1,436). happensAt(walking(lb1gt_1),436). orientation(lb1gt_2,144,436). coords(lb1gt_2,189,142,436).  visible(lb1gt_2,436). happensAt(active(lb1gt_2),436).
orientation(lb1gt_1,174,437). coords(lb1gt_1,170,188,437).  visible(lb1gt_1,437). happensAt(walking(lb1gt_1),437). orientation(lb1gt_2,144,437). coords(lb1gt_2,189,142,437).  visible(lb1gt_2,437). happensAt(walking(lb1gt_2),437).
orientation(lb1gt_1,164,438). coords(lb1gt_1,171,191,438).  visible(lb1gt_1,438). happensAt(walking(lb1gt_1),438). orientation(lb1gt_2,144,438). coords(lb1gt_2,189,142,438).  visible(lb1gt_2,438). happensAt(walking(lb1gt_2),438).
orientation(lb1gt_1,162,439). coords(lb1gt_1,172,193,439).  visible(lb1gt_1,439). happensAt(walking(lb1gt_1),439). orientation(lb1gt_2,144,439). coords(lb1gt_2,188,143,439).  visible(lb1gt_2,439). happensAt(walking(lb1gt_2),439).
orientation(lb1gt_1,159,440). coords(lb1gt_1,172,195,440).  visible(lb1gt_1,440). happensAt(walking(lb1gt_1),440). orientation(lb1gt_2,144,440). coords(lb1gt_2,188,143,440).  visible(lb1gt_2,440). happensAt(walking(lb1gt_2),440).
orientation(lb1gt_1,175,441). coords(lb1gt_1,173,196,441).  visible(lb1gt_1,441). happensAt(walking(lb1gt_1),441). orientation(lb1gt_2,144,441). coords(lb1gt_2,188,143,441).  visible(lb1gt_2,441). happensAt(walking(lb1gt_2),441).
orientation(lb1gt_1,174,442). coords(lb1gt_1,174,197,442).  visible(lb1gt_1,442). happensAt(walking(lb1gt_1),442). orientation(lb1gt_2,146,442). coords(lb1gt_2,188,145,442).  visible(lb1gt_2,442). happensAt(walking(lb1gt_2),442).
orientation(lb1gt_1,177,443). coords(lb1gt_1,175,198,443).  visible(lb1gt_1,443). happensAt(walking(lb1gt_1),443). orientation(lb1gt_2,146,443). coords(lb1gt_2,187,146,443).  visible(lb1gt_2,443). happensAt(walking(lb1gt_2),443).
orientation(lb1gt_1,3,444). coords(lb1gt_1,175,198,444).  visible(lb1gt_1,444). happensAt(walking(lb1gt_1),444). orientation(lb1gt_2,146,444). coords(lb1gt_2,187,148,444).  visible(lb1gt_2,444). happensAt(walking(lb1gt_2),444).
orientation(lb1gt_1,1,445). coords(lb1gt_1,176,199,445).  visible(lb1gt_1,445). happensAt(walking(lb1gt_1),445). orientation(lb1gt_2,146,445). coords(lb1gt_2,187,150,445).  visible(lb1gt_2,445). happensAt(walking(lb1gt_2),445).
orientation(lb1gt_1,2,446). coords(lb1gt_1,176,199,446).  visible(lb1gt_1,446). happensAt(walking(lb1gt_1),446). orientation(lb1gt_2,146,446). coords(lb1gt_2,187,151,446).  visible(lb1gt_2,446). happensAt(walking(lb1gt_2),446).
orientation(lb1gt_1,4,447). coords(lb1gt_1,177,200,447).  visible(lb1gt_1,447). happensAt(walking(lb1gt_1),447). orientation(lb1gt_2,146,447). coords(lb1gt_2,187,153,447).  visible(lb1gt_2,447). happensAt(walking(lb1gt_2),447).
orientation(lb1gt_1,0,448). coords(lb1gt_1,178,201,448).  visible(lb1gt_1,448). happensAt(walking(lb1gt_1),448). orientation(lb1gt_2,146,448). coords(lb1gt_2,186,154,448).  visible(lb1gt_2,448). happensAt(walking(lb1gt_2),448).
orientation(lb1gt_1,0,449). coords(lb1gt_1,178,202,449).  visible(lb1gt_1,449). happensAt(walking(lb1gt_1),449). orientation(lb1gt_2,146,449). coords(lb1gt_2,187,155,449).  visible(lb1gt_2,449). happensAt(walking(lb1gt_2),449).
orientation(lb1gt_1,172,450). coords(lb1gt_1,178,204,450).  visible(lb1gt_1,450). happensAt(walking(lb1gt_1),450). orientation(lb1gt_2,146,450). coords(lb1gt_2,187,156,450).  visible(lb1gt_2,450). happensAt(walking(lb1gt_2),450).
orientation(lb1gt_1,175,451). coords(lb1gt_1,178,206,451).  visible(lb1gt_1,451). happensAt(walking(lb1gt_1),451). orientation(lb1gt_2,146,451). coords(lb1gt_2,187,156,451).  visible(lb1gt_2,451). happensAt(walking(lb1gt_2),451).
orientation(lb1gt_1,0,452). coords(lb1gt_1,179,208,452).  visible(lb1gt_1,452). happensAt(walking(lb1gt_1),452). orientation(lb1gt_2,146,452). coords(lb1gt_2,187,157,452).  visible(lb1gt_2,452). happensAt(walking(lb1gt_2),452).
orientation(lb1gt_1,0,453). coords(lb1gt_1,179,209,453).  visible(lb1gt_1,453). happensAt(walking(lb1gt_1),453). orientation(lb1gt_2,146,453). coords(lb1gt_2,187,158,453).  visible(lb1gt_2,453). happensAt(walking(lb1gt_2),453).
orientation(lb1gt_1,178,454). coords(lb1gt_1,179,211,454).  visible(lb1gt_1,454). happensAt(walking(lb1gt_1),454). orientation(lb1gt_2,146,454). coords(lb1gt_2,187,159,454).  visible(lb1gt_2,454). happensAt(walking(lb1gt_2),454).
orientation(lb1gt_1,178,455). coords(lb1gt_1,180,213,455).  visible(lb1gt_1,455). happensAt(walking(lb1gt_1),455). orientation(lb1gt_2,146,455). coords(lb1gt_2,187,159,455).  visible(lb1gt_2,455). happensAt(walking(lb1gt_2),455).
orientation(lb1gt_1,176,456). coords(lb1gt_1,181,217,456).  visible(lb1gt_1,456). happensAt(walking(lb1gt_1),456). orientation(lb1gt_2,146,456). coords(lb1gt_2,187,160,456).  visible(lb1gt_2,456). happensAt(walking(lb1gt_2),456).
#show holdsAt/2.