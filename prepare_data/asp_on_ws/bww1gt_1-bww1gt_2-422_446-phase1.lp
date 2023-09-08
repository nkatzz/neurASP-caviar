
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

	orientation(bww1gt_1,2,422). coords(bww1gt_1,105,264,422).  visible(bww1gt_1,422). happensAt(inactive(bww1gt_1),422). orientation(bww1gt_2,154,422). coords(bww1gt_2,133,145,422).  visible(bww1gt_2,422). happensAt(walking(bww1gt_2),422).
orientation(bww1gt_1,2,423). coords(bww1gt_1,105,264,423).  visible(bww1gt_1,423). happensAt(inactive(bww1gt_1),423). orientation(bww1gt_2,156,423). coords(bww1gt_2,133,145,423).  visible(bww1gt_2,423). happensAt(walking(bww1gt_2),423).
orientation(bww1gt_1,3,424). coords(bww1gt_1,105,264,424).  visible(bww1gt_1,424). happensAt(inactive(bww1gt_1),424). orientation(bww1gt_2,155,424). coords(bww1gt_2,134,144,424).  visible(bww1gt_2,424). happensAt(walking(bww1gt_2),424).
orientation(bww1gt_1,1,425). coords(bww1gt_1,105,264,425).  visible(bww1gt_1,425). happensAt(inactive(bww1gt_1),425). orientation(bww1gt_2,153,425). coords(bww1gt_2,134,144,425).  visible(bww1gt_2,425). happensAt(walking(bww1gt_2),425).
orientation(bww1gt_1,2,426). coords(bww1gt_1,105,264,426).  visible(bww1gt_1,426). happensAt(inactive(bww1gt_1),426). orientation(bww1gt_2,152,426). coords(bww1gt_2,134,144,426).  visible(bww1gt_2,426). happensAt(walking(bww1gt_2),426).
orientation(bww1gt_1,179,427). coords(bww1gt_1,105,264,427).  visible(bww1gt_1,427). happensAt(inactive(bww1gt_1),427). orientation(bww1gt_2,153,427). coords(bww1gt_2,134,144,427).  visible(bww1gt_2,427). happensAt(walking(bww1gt_2),427).
orientation(bww1gt_1,3,428). coords(bww1gt_1,105,264,428).  visible(bww1gt_1,428). happensAt(inactive(bww1gt_1),428). orientation(bww1gt_2,152,428). coords(bww1gt_2,135,144,428).  visible(bww1gt_2,428). happensAt(walking(bww1gt_2),428).
orientation(bww1gt_1,4,429). coords(bww1gt_1,105,264,429).  visible(bww1gt_1,429). happensAt(inactive(bww1gt_1),429). orientation(bww1gt_2,151,429). coords(bww1gt_2,135,144,429).  visible(bww1gt_2,429). happensAt(walking(bww1gt_2),429).
orientation(bww1gt_1,5,430). coords(bww1gt_1,105,264,430).  visible(bww1gt_1,430). happensAt(inactive(bww1gt_1),430). orientation(bww1gt_2,153,430). coords(bww1gt_2,136,144,430).  visible(bww1gt_2,430). happensAt(walking(bww1gt_2),430).
orientation(bww1gt_1,1,431). coords(bww1gt_1,105,264,431).  visible(bww1gt_1,431). happensAt(inactive(bww1gt_1),431). orientation(bww1gt_2,155,431). coords(bww1gt_2,138,145,431).  visible(bww1gt_2,431). happensAt(walking(bww1gt_2),431).
orientation(bww1gt_1,7,432). coords(bww1gt_1,105,264,432).  visible(bww1gt_1,432). happensAt(inactive(bww1gt_1),432). orientation(bww1gt_2,153,432). coords(bww1gt_2,140,145,432).  visible(bww1gt_2,432). happensAt(walking(bww1gt_2),432).
orientation(bww1gt_1,2,433). coords(bww1gt_1,105,264,433).  visible(bww1gt_1,433). happensAt(inactive(bww1gt_1),433). orientation(bww1gt_2,149,433). coords(bww1gt_2,141,145,433).  visible(bww1gt_2,433). happensAt(walking(bww1gt_2),433).
orientation(bww1gt_1,178,434). coords(bww1gt_1,105,264,434).  visible(bww1gt_1,434). happensAt(inactive(bww1gt_1),434). orientation(bww1gt_2,151,434). coords(bww1gt_2,142,146,434).  visible(bww1gt_2,434). happensAt(walking(bww1gt_2),434).
orientation(bww1gt_1,0,435). coords(bww1gt_1,105,264,435).  visible(bww1gt_1,435). happensAt(inactive(bww1gt_1),435). orientation(bww1gt_2,153,435). coords(bww1gt_2,144,146,435).  visible(bww1gt_2,435). happensAt(walking(bww1gt_2),435).
orientation(bww1gt_1,0,436). coords(bww1gt_1,105,264,436).  visible(bww1gt_1,436). happensAt(inactive(bww1gt_1),436). orientation(bww1gt_2,153,436). coords(bww1gt_2,144,146,436).  visible(bww1gt_2,436). happensAt(walking(bww1gt_2),436).
orientation(bww1gt_1,0,437). coords(bww1gt_1,105,264,437).  visible(bww1gt_1,437). happensAt(inactive(bww1gt_1),437). orientation(bww1gt_2,156,437). coords(bww1gt_2,145,146,437).  visible(bww1gt_2,437). happensAt(walking(bww1gt_2),437).
orientation(bww1gt_1,0,438). coords(bww1gt_1,105,264,438).  visible(bww1gt_1,438). happensAt(inactive(bww1gt_1),438). orientation(bww1gt_2,156,438). coords(bww1gt_2,146,146,438).  visible(bww1gt_2,438). happensAt(walking(bww1gt_2),438).
orientation(bww1gt_1,0,439). coords(bww1gt_1,105,264,439).  visible(bww1gt_1,439). happensAt(inactive(bww1gt_1),439). orientation(bww1gt_2,155,439). coords(bww1gt_2,146,147,439).  visible(bww1gt_2,439). happensAt(walking(bww1gt_2),439).
orientation(bww1gt_1,0,440). coords(bww1gt_1,105,264,440).  visible(bww1gt_1,440). happensAt(inactive(bww1gt_1),440). orientation(bww1gt_2,153,440). coords(bww1gt_2,146,147,440).  visible(bww1gt_2,440). happensAt(walking(bww1gt_2),440).
orientation(bww1gt_1,0,441). coords(bww1gt_1,105,264,441).  visible(bww1gt_1,441). happensAt(inactive(bww1gt_1),441). orientation(bww1gt_2,155,441). coords(bww1gt_2,146,147,441).  visible(bww1gt_2,441). happensAt(walking(bww1gt_2),441).
orientation(bww1gt_1,0,442). coords(bww1gt_1,105,264,442).  visible(bww1gt_1,442). happensAt(inactive(bww1gt_1),442). orientation(bww1gt_2,153,442). coords(bww1gt_2,146,147,442).  visible(bww1gt_2,442). happensAt(walking(bww1gt_2),442).
orientation(bww1gt_1,0,443). coords(bww1gt_1,105,264,443).  visible(bww1gt_1,443). happensAt(inactive(bww1gt_1),443). orientation(bww1gt_2,152,443). coords(bww1gt_2,148,147,443).  visible(bww1gt_2,443). happensAt(walking(bww1gt_2),443).
orientation(bww1gt_1,0,444). coords(bww1gt_1,105,264,444).  visible(bww1gt_1,444). happensAt(inactive(bww1gt_1),444). orientation(bww1gt_2,152,444). coords(bww1gt_2,148,147,444).  visible(bww1gt_2,444). happensAt(walking(bww1gt_2),444).
orientation(bww1gt_1,0,445). coords(bww1gt_1,105,264,445).  visible(bww1gt_1,445). happensAt(inactive(bww1gt_1),445). orientation(bww1gt_2,153,445). coords(bww1gt_2,148,147,445).  visible(bww1gt_2,445). happensAt(walking(bww1gt_2),445).
orientation(bww1gt_1,0,446). coords(bww1gt_1,105,264,446).  visible(bww1gt_1,446). happensAt(inactive(bww1gt_1),446). orientation(bww1gt_2,155,446). coords(bww1gt_2,151,147,446).  visible(bww1gt_2,446). happensAt(walking(bww1gt_2),446).
#show holdsAt/2.