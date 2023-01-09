
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

	orientation(mwt1gt_1,160,425). coords(mwt1gt_1,80,108,425).  visible(mwt1gt_1,425). happensAt(walking(mwt1gt_1),425). orientation(mwt1gt_2,140,425). coords(mwt1gt_2,81,95,425).  visible(mwt1gt_2,425). happensAt(walking(mwt1gt_2),425).
orientation(mwt1gt_1,160,426). coords(mwt1gt_1,80,108,426).  visible(mwt1gt_1,426). happensAt(walking(mwt1gt_1),426). orientation(mwt1gt_2,140,426). coords(mwt1gt_2,81,95,426).  visible(mwt1gt_2,426). happensAt(walking(mwt1gt_2),426).
orientation(mwt1gt_1,160,427). coords(mwt1gt_1,79,106,427).  visible(mwt1gt_1,427). happensAt(walking(mwt1gt_1),427). orientation(mwt1gt_2,140,427). coords(mwt1gt_2,80,94,427).  visible(mwt1gt_2,427). happensAt(walking(mwt1gt_2),427).
orientation(mwt1gt_1,160,428). coords(mwt1gt_1,79,106,428).  visible(mwt1gt_1,428). happensAt(walking(mwt1gt_1),428). orientation(mwt1gt_2,140,428). coords(mwt1gt_2,80,94,428).  visible(mwt1gt_2,428). happensAt(walking(mwt1gt_2),428).
orientation(mwt1gt_1,160,429). coords(mwt1gt_1,79,106,429).  visible(mwt1gt_1,429). happensAt(walking(mwt1gt_1),429). orientation(mwt1gt_2,140,429). coords(mwt1gt_2,80,94,429).  visible(mwt1gt_2,429). happensAt(walking(mwt1gt_2),429).
orientation(mwt1gt_1,160,430). coords(mwt1gt_1,79,106,430).  visible(mwt1gt_1,430). happensAt(walking(mwt1gt_1),430). orientation(mwt1gt_2,140,430). coords(mwt1gt_2,80,94,430).  visible(mwt1gt_2,430). happensAt(walking(mwt1gt_2),430).
orientation(mwt1gt_1,160,431). coords(mwt1gt_1,78,105,431).  visible(mwt1gt_1,431). happensAt(walking(mwt1gt_1),431). orientation(mwt1gt_2,140,431). coords(mwt1gt_2,80,94,431).  visible(mwt1gt_2,431). happensAt(walking(mwt1gt_2),431).
orientation(mwt1gt_1,160,432). coords(mwt1gt_1,78,105,432).  visible(mwt1gt_1,432). happensAt(walking(mwt1gt_1),432). orientation(mwt1gt_2,140,432). coords(mwt1gt_2,80,94,432).  visible(mwt1gt_2,432). happensAt(walking(mwt1gt_2),432).
orientation(mwt1gt_1,160,433). coords(mwt1gt_1,78,105,433).  visible(mwt1gt_1,433). happensAt(walking(mwt1gt_1),433). orientation(mwt1gt_2,140,433). coords(mwt1gt_2,80,94,433).  visible(mwt1gt_2,433). happensAt(walking(mwt1gt_2),433).
orientation(mwt1gt_1,160,434). coords(mwt1gt_1,74,104,434).  visible(mwt1gt_1,434). happensAt(walking(mwt1gt_1),434). orientation(mwt1gt_2,140,434). coords(mwt1gt_2,75,92,434).  visible(mwt1gt_2,434). happensAt(walking(mwt1gt_2),434).
orientation(mwt1gt_1,160,435). coords(mwt1gt_1,74,104,435).  visible(mwt1gt_1,435). happensAt(walking(mwt1gt_1),435). orientation(mwt1gt_2,140,435). coords(mwt1gt_2,75,92,435).  visible(mwt1gt_2,435). happensAt(walking(mwt1gt_2),435).
orientation(mwt1gt_1,160,436). coords(mwt1gt_1,74,104,436).  visible(mwt1gt_1,436). happensAt(walking(mwt1gt_1),436). orientation(mwt1gt_2,140,436). coords(mwt1gt_2,75,92,436).  visible(mwt1gt_2,436). happensAt(walking(mwt1gt_2),436).
orientation(mwt1gt_1,160,437). coords(mwt1gt_1,74,104,437).  visible(mwt1gt_1,437). happensAt(walking(mwt1gt_1),437). orientation(mwt1gt_2,140,437). coords(mwt1gt_2,75,92,437).  visible(mwt1gt_2,437). happensAt(walking(mwt1gt_2),437).
orientation(mwt1gt_1,160,438). coords(mwt1gt_1,74,104,438).  visible(mwt1gt_1,438). happensAt(walking(mwt1gt_1),438). orientation(mwt1gt_2,140,438). coords(mwt1gt_2,75,92,438).  visible(mwt1gt_2,438). happensAt(walking(mwt1gt_2),438).
orientation(mwt1gt_1,160,439). coords(mwt1gt_1,74,104,439).  visible(mwt1gt_1,439). happensAt(walking(mwt1gt_1),439). orientation(mwt1gt_2,140,439). coords(mwt1gt_2,75,92,439).  visible(mwt1gt_2,439). happensAt(walking(mwt1gt_2),439).
orientation(mwt1gt_1,160,440). coords(mwt1gt_1,74,104,440).  visible(mwt1gt_1,440). happensAt(walking(mwt1gt_1),440). orientation(mwt1gt_2,140,440). coords(mwt1gt_2,75,92,440).  visible(mwt1gt_2,440). happensAt(walking(mwt1gt_2),440).
orientation(mwt1gt_1,160,441). coords(mwt1gt_1,74,104,441).  visible(mwt1gt_1,441). happensAt(walking(mwt1gt_1),441). orientation(mwt1gt_2,140,441). coords(mwt1gt_2,75,92,441).  visible(mwt1gt_2,441). happensAt(walking(mwt1gt_2),441).
orientation(mwt1gt_1,160,442). coords(mwt1gt_1,74,104,442).  visible(mwt1gt_1,442). happensAt(walking(mwt1gt_1),442). orientation(mwt1gt_2,140,442). coords(mwt1gt_2,75,92,442).  visible(mwt1gt_2,442). happensAt(walking(mwt1gt_2),442).
orientation(mwt1gt_1,160,443). coords(mwt1gt_1,74,104,443).  visible(mwt1gt_1,443). happensAt(walking(mwt1gt_1),443). orientation(mwt1gt_2,140,443). coords(mwt1gt_2,75,92,443).  visible(mwt1gt_2,443). happensAt(walking(mwt1gt_2),443).
orientation(mwt1gt_1,160,444). coords(mwt1gt_1,74,104,444).  visible(mwt1gt_1,444). happensAt(walking(mwt1gt_1),444). orientation(mwt1gt_2,140,444). coords(mwt1gt_2,75,92,444).  visible(mwt1gt_2,444). happensAt(walking(mwt1gt_2),444).
orientation(mwt1gt_1,160,445). coords(mwt1gt_1,74,101,445).  visible(mwt1gt_1,445). happensAt(walking(mwt1gt_1),445). orientation(mwt1gt_2,140,445). coords(mwt1gt_2,74,90,445).  visible(mwt1gt_2,445). happensAt(walking(mwt1gt_2),445).
orientation(mwt1gt_1,160,446). coords(mwt1gt_1,74,101,446).  visible(mwt1gt_1,446). happensAt(walking(mwt1gt_1),446). orientation(mwt1gt_2,140,446). coords(mwt1gt_2,74,90,446).  visible(mwt1gt_2,446). happensAt(walking(mwt1gt_2),446).
orientation(mwt1gt_1,160,447). coords(mwt1gt_1,72,99,447).  visible(mwt1gt_1,447). happensAt(walking(mwt1gt_1),447). orientation(mwt1gt_2,140,447). coords(mwt1gt_2,72,88,447).  visible(mwt1gt_2,447). happensAt(walking(mwt1gt_2),447).
orientation(mwt1gt_1,160,448). coords(mwt1gt_1,72,99,448).  visible(mwt1gt_1,448). happensAt(walking(mwt1gt_1),448). orientation(mwt1gt_2,140,448). coords(mwt1gt_2,72,88,448).  visible(mwt1gt_2,448). happensAt(walking(mwt1gt_2),448).
orientation(mwt1gt_1,160,449). coords(mwt1gt_1,72,99,449).  visible(mwt1gt_1,449). happensAt(walking(mwt1gt_1),449). orientation(mwt1gt_2,140,449). coords(mwt1gt_2,72,88,449).  visible(mwt1gt_2,449). happensAt(walking(mwt1gt_2),449).
#show holdsAt/2.