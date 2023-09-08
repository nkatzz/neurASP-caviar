
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

	orientation(lb1gt_1,151,420). coords(lb1gt_1,166,175,420).  visible(lb1gt_1,420). happensAt(walking(lb1gt_1),420). orientation(lb1gt_2,144,420). coords(lb1gt_2,189,141,420).  visible(lb1gt_2,420). happensAt(active(lb1gt_2),420).
orientation(lb1gt_1,151,421). coords(lb1gt_1,166,176,421).  visible(lb1gt_1,421). happensAt(walking(lb1gt_1),421). orientation(lb1gt_2,144,421). coords(lb1gt_2,189,141,421).  visible(lb1gt_2,421). happensAt(active(lb1gt_2),421).
orientation(lb1gt_1,151,422). coords(lb1gt_1,166,177,422).  visible(lb1gt_1,422). happensAt(walking(lb1gt_1),422). orientation(lb1gt_2,144,422). coords(lb1gt_2,189,141,422).  visible(lb1gt_2,422). happensAt(active(lb1gt_2),422).
orientation(lb1gt_1,151,423). coords(lb1gt_1,166,178,423).  visible(lb1gt_1,423). happensAt(walking(lb1gt_1),423). orientation(lb1gt_2,144,423). coords(lb1gt_2,189,141,423).  visible(lb1gt_2,423). happensAt(active(lb1gt_2),423).
orientation(lb1gt_1,151,424). coords(lb1gt_1,166,179,424).  visible(lb1gt_1,424). happensAt(walking(lb1gt_1),424). orientation(lb1gt_2,144,424). coords(lb1gt_2,189,141,424).  visible(lb1gt_2,424). happensAt(active(lb1gt_2),424).
orientation(lb1gt_1,161,425). coords(lb1gt_1,166,179,425).  visible(lb1gt_1,425). happensAt(walking(lb1gt_1),425). orientation(lb1gt_2,144,425). coords(lb1gt_2,189,141,425).  visible(lb1gt_2,425). happensAt(active(lb1gt_2),425).
orientation(lb1gt_1,162,426). coords(lb1gt_1,166,180,426).  visible(lb1gt_1,426). happensAt(walking(lb1gt_1),426). orientation(lb1gt_2,144,426). coords(lb1gt_2,189,141,426).  visible(lb1gt_2,426). happensAt(active(lb1gt_2),426).
orientation(lb1gt_1,162,427). coords(lb1gt_1,166,180,427).  visible(lb1gt_1,427). happensAt(walking(lb1gt_1),427). orientation(lb1gt_2,144,427). coords(lb1gt_2,189,141,427).  visible(lb1gt_2,427). happensAt(active(lb1gt_2),427).
orientation(lb1gt_1,163,428). coords(lb1gt_1,165,181,428).  visible(lb1gt_1,428). happensAt(walking(lb1gt_1),428). orientation(lb1gt_2,144,428). coords(lb1gt_2,189,141,428).  visible(lb1gt_2,428). happensAt(active(lb1gt_2),428).
orientation(lb1gt_1,162,429). coords(lb1gt_1,165,181,429).  visible(lb1gt_1,429). happensAt(walking(lb1gt_1),429). orientation(lb1gt_2,144,429). coords(lb1gt_2,189,141,429).  visible(lb1gt_2,429). happensAt(active(lb1gt_2),429).
orientation(lb1gt_1,162,430). coords(lb1gt_1,165,182,430).  visible(lb1gt_1,430). happensAt(walking(lb1gt_1),430). orientation(lb1gt_2,144,430). coords(lb1gt_2,189,141,430).  visible(lb1gt_2,430). happensAt(active(lb1gt_2),430).
orientation(lb1gt_1,163,431). coords(lb1gt_1,165,182,431).  visible(lb1gt_1,431). happensAt(walking(lb1gt_1),431). orientation(lb1gt_2,144,431). coords(lb1gt_2,189,141,431).  visible(lb1gt_2,431). happensAt(active(lb1gt_2),431).
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
#show holdsAt/2.