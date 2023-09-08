
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

	orientation(wk1gt_4,157,423). coords(wk1gt_4,54,174,423).  visible(wk1gt_4,423). happensAt(active(wk1gt_4),423). orientation(wk1gt_5,162,423). coords(wk1gt_5,50,163,423).  visible(wk1gt_5,423). happensAt(active(wk1gt_5),423).
orientation(wk1gt_4,157,424). coords(wk1gt_4,54,174,424).  visible(wk1gt_4,424). happensAt(active(wk1gt_4),424). orientation(wk1gt_5,162,424). coords(wk1gt_5,50,163,424).  visible(wk1gt_5,424). happensAt(active(wk1gt_5),424).
orientation(wk1gt_4,157,425). coords(wk1gt_4,54,173,425).  visible(wk1gt_4,425). happensAt(active(wk1gt_4),425). orientation(wk1gt_5,162,425). coords(wk1gt_5,50,163,425).  visible(wk1gt_5,425). happensAt(active(wk1gt_5),425).
orientation(wk1gt_4,147,426). coords(wk1gt_4,54,173,426).  visible(wk1gt_4,426). happensAt(active(wk1gt_4),426). orientation(wk1gt_5,162,426). coords(wk1gt_5,50,163,426).  visible(wk1gt_5,426). happensAt(active(wk1gt_5),426).
orientation(wk1gt_4,147,427). coords(wk1gt_4,54,172,427).  visible(wk1gt_4,427). happensAt(active(wk1gt_4),427). orientation(wk1gt_5,162,427). coords(wk1gt_5,50,163,427).  visible(wk1gt_5,427). happensAt(active(wk1gt_5),427).
orientation(wk1gt_4,147,428). coords(wk1gt_4,54,172,428).  visible(wk1gt_4,428). happensAt(active(wk1gt_4),428). orientation(wk1gt_5,162,428). coords(wk1gt_5,50,163,428).  visible(wk1gt_5,428). happensAt(active(wk1gt_5),428).
orientation(wk1gt_4,147,429). coords(wk1gt_4,54,172,429).  visible(wk1gt_4,429). happensAt(active(wk1gt_4),429). orientation(wk1gt_5,162,429). coords(wk1gt_5,50,163,429).  visible(wk1gt_5,429). happensAt(active(wk1gt_5),429).
orientation(wk1gt_4,147,430). coords(wk1gt_4,54,171,430).  visible(wk1gt_4,430). happensAt(active(wk1gt_4),430). orientation(wk1gt_5,162,430). coords(wk1gt_5,50,163,430).  visible(wk1gt_5,430). happensAt(active(wk1gt_5),430).
orientation(wk1gt_4,147,431). coords(wk1gt_4,54,171,431).  visible(wk1gt_4,431). happensAt(active(wk1gt_4),431). orientation(wk1gt_5,162,431). coords(wk1gt_5,50,163,431).  visible(wk1gt_5,431). happensAt(active(wk1gt_5),431).
orientation(wk1gt_4,147,432). coords(wk1gt_4,54,171,432).  visible(wk1gt_4,432). happensAt(active(wk1gt_4),432). orientation(wk1gt_5,162,432). coords(wk1gt_5,50,163,432).  visible(wk1gt_5,432). happensAt(active(wk1gt_5),432).
orientation(wk1gt_4,147,433). coords(wk1gt_4,54,171,433).  visible(wk1gt_4,433). happensAt(active(wk1gt_4),433). orientation(wk1gt_5,162,433). coords(wk1gt_5,50,163,433).  visible(wk1gt_5,433). happensAt(active(wk1gt_5),433).
orientation(wk1gt_4,147,434). coords(wk1gt_4,54,170,434).  visible(wk1gt_4,434). happensAt(active(wk1gt_4),434). orientation(wk1gt_5,162,434). coords(wk1gt_5,51,162,434).  visible(wk1gt_5,434). happensAt(active(wk1gt_5),434).
orientation(wk1gt_4,147,435). coords(wk1gt_4,55,170,435).  visible(wk1gt_4,435). happensAt(active(wk1gt_4),435). orientation(wk1gt_5,162,435). coords(wk1gt_5,51,162,435).  visible(wk1gt_5,435). happensAt(active(wk1gt_5),435).
orientation(wk1gt_4,147,436). coords(wk1gt_4,55,169,436).  visible(wk1gt_4,436). happensAt(active(wk1gt_4),436). orientation(wk1gt_5,162,436). coords(wk1gt_5,52,160,436).  visible(wk1gt_5,436). happensAt(active(wk1gt_5),436).
orientation(wk1gt_4,147,437). coords(wk1gt_4,56,169,437).  visible(wk1gt_4,437). happensAt(active(wk1gt_4),437). orientation(wk1gt_5,162,437). coords(wk1gt_5,52,160,437).  visible(wk1gt_5,437). happensAt(active(wk1gt_5),437).
orientation(wk1gt_4,147,438). coords(wk1gt_4,56,169,438).  visible(wk1gt_4,438). happensAt(active(wk1gt_4),438). orientation(wk1gt_5,162,438). coords(wk1gt_5,52,160,438).  visible(wk1gt_5,438). happensAt(active(wk1gt_5),438).
orientation(wk1gt_4,147,439). coords(wk1gt_4,56,169,439).  visible(wk1gt_4,439). happensAt(active(wk1gt_4),439). orientation(wk1gt_5,162,439). coords(wk1gt_5,52,160,439).  visible(wk1gt_5,439). happensAt(active(wk1gt_5),439).
orientation(wk1gt_4,147,440). coords(wk1gt_4,56,169,440).  visible(wk1gt_4,440). happensAt(active(wk1gt_4),440). orientation(wk1gt_5,162,440). coords(wk1gt_5,52,160,440).  visible(wk1gt_5,440). happensAt(active(wk1gt_5),440).
orientation(wk1gt_4,147,441). coords(wk1gt_4,56,169,441).  visible(wk1gt_4,441). happensAt(active(wk1gt_4),441). orientation(wk1gt_5,162,441). coords(wk1gt_5,53,160,441).  visible(wk1gt_5,441). happensAt(active(wk1gt_5),441).
orientation(wk1gt_4,147,442). coords(wk1gt_4,56,169,442).  visible(wk1gt_4,442). happensAt(active(wk1gt_4),442). orientation(wk1gt_5,162,442). coords(wk1gt_5,54,160,442).  visible(wk1gt_5,442). happensAt(active(wk1gt_5),442).
orientation(wk1gt_4,147,443). coords(wk1gt_4,56,169,443).  visible(wk1gt_4,443). happensAt(active(wk1gt_4),443). orientation(wk1gt_5,162,443). coords(wk1gt_5,54,160,443).  visible(wk1gt_5,443). happensAt(active(wk1gt_5),443).
orientation(wk1gt_4,147,444). coords(wk1gt_4,56,169,444).  visible(wk1gt_4,444). happensAt(active(wk1gt_4),444). orientation(wk1gt_5,162,444). coords(wk1gt_5,54,160,444).  visible(wk1gt_5,444). happensAt(walking(wk1gt_5),444).
orientation(wk1gt_4,147,445). coords(wk1gt_4,56,169,445).  visible(wk1gt_4,445). happensAt(active(wk1gt_4),445). orientation(wk1gt_5,162,445). coords(wk1gt_5,54,160,445).  visible(wk1gt_5,445). happensAt(walking(wk1gt_5),445).
orientation(wk1gt_4,147,446). coords(wk1gt_4,56,169,446).  visible(wk1gt_4,446). happensAt(active(wk1gt_4),446). orientation(wk1gt_5,162,446). coords(wk1gt_5,54,159,446).  visible(wk1gt_5,446). happensAt(walking(wk1gt_5),446).
orientation(wk1gt_4,147,447). coords(wk1gt_4,56,169,447).  visible(wk1gt_4,447). happensAt(active(wk1gt_4),447). orientation(wk1gt_5,162,447). coords(wk1gt_5,54,159,447).  visible(wk1gt_5,447). happensAt(walking(wk1gt_5),447).
#show holdsAt/2.