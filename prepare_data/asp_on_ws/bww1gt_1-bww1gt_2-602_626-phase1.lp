
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

	orientation(bww1gt_1,173,602). coords(bww1gt_1,101,250,602).  visible(bww1gt_1,602). happensAt(walking(bww1gt_1),602). orientation(bww1gt_2,0,602). coords(bww1gt_2,134,211,602).  visible(bww1gt_2,602). happensAt(active(bww1gt_2),602).
orientation(bww1gt_1,173,603). coords(bww1gt_1,101,250,603).  visible(bww1gt_1,603). happensAt(walking(bww1gt_1),603). orientation(bww1gt_2,178,603). coords(bww1gt_2,134,211,603).  visible(bww1gt_2,603). happensAt(active(bww1gt_2),603).
orientation(bww1gt_1,173,604). coords(bww1gt_1,101,250,604).  visible(bww1gt_1,604). happensAt(walking(bww1gt_1),604). orientation(bww1gt_2,1,604). coords(bww1gt_2,134,211,604).  visible(bww1gt_2,604). happensAt(active(bww1gt_2),604).
orientation(bww1gt_1,173,605). coords(bww1gt_1,101,250,605).  visible(bww1gt_1,605). happensAt(walking(bww1gt_1),605). orientation(bww1gt_2,178,605). coords(bww1gt_2,135,211,605).  visible(bww1gt_2,605). happensAt(active(bww1gt_2),605).
orientation(bww1gt_1,173,606). coords(bww1gt_1,101,250,606).  visible(bww1gt_1,606). happensAt(walking(bww1gt_1),606). orientation(bww1gt_2,179,606). coords(bww1gt_2,135,211,606).  visible(bww1gt_2,606). happensAt(active(bww1gt_2),606).
orientation(bww1gt_1,173,607). coords(bww1gt_1,101,250,607).  visible(bww1gt_1,607). happensAt(walking(bww1gt_1),607). orientation(bww1gt_2,176,607). coords(bww1gt_2,135,211,607).  visible(bww1gt_2,607). happensAt(active(bww1gt_2),607).
orientation(bww1gt_1,173,608). coords(bww1gt_1,101,250,608).  visible(bww1gt_1,608). happensAt(walking(bww1gt_1),608). orientation(bww1gt_2,177,608). coords(bww1gt_2,135,212,608).  visible(bww1gt_2,608). happensAt(active(bww1gt_2),608).
orientation(bww1gt_1,173,609). coords(bww1gt_1,101,250,609).  visible(bww1gt_1,609). happensAt(walking(bww1gt_1),609). orientation(bww1gt_2,177,609). coords(bww1gt_2,136,212,609).  visible(bww1gt_2,609). happensAt(active(bww1gt_2),609).
orientation(bww1gt_1,173,610). coords(bww1gt_1,101,251,610).  visible(bww1gt_1,610). happensAt(walking(bww1gt_1),610). orientation(bww1gt_2,1,610). coords(bww1gt_2,137,212,610).  visible(bww1gt_2,610). happensAt(active(bww1gt_2),610).
orientation(bww1gt_1,173,611). coords(bww1gt_1,101,251,611).  visible(bww1gt_1,611). happensAt(walking(bww1gt_1),611). orientation(bww1gt_2,2,611). coords(bww1gt_2,138,211,611).  visible(bww1gt_2,611). happensAt(active(bww1gt_2),611).
orientation(bww1gt_1,173,612). coords(bww1gt_1,101,251,612).  visible(bww1gt_1,612). happensAt(walking(bww1gt_1),612). orientation(bww1gt_2,179,612). coords(bww1gt_2,139,211,612).  visible(bww1gt_2,612). happensAt(active(bww1gt_2),612).
orientation(bww1gt_1,173,613). coords(bww1gt_1,101,251,613).  visible(bww1gt_1,613). happensAt(walking(bww1gt_1),613). orientation(bww1gt_2,1,613). coords(bww1gt_2,139,210,613).  visible(bww1gt_2,613). happensAt(active(bww1gt_2),613).
orientation(bww1gt_1,173,614). coords(bww1gt_1,101,251,614).  visible(bww1gt_1,614). happensAt(walking(bww1gt_1),614). orientation(bww1gt_2,2,614). coords(bww1gt_2,140,209,614).  visible(bww1gt_2,614). happensAt(active(bww1gt_2),614).
orientation(bww1gt_1,173,615). coords(bww1gt_1,101,252,615).  visible(bww1gt_1,615). happensAt(walking(bww1gt_1),615). orientation(bww1gt_2,4,615). coords(bww1gt_2,140,208,615).  visible(bww1gt_2,615). happensAt(active(bww1gt_2),615).
orientation(bww1gt_1,173,616). coords(bww1gt_1,101,252,616).  visible(bww1gt_1,616). happensAt(walking(bww1gt_1),616). orientation(bww1gt_2,3,616). coords(bww1gt_2,141,207,616).  visible(bww1gt_2,616). happensAt(active(bww1gt_2),616).
orientation(bww1gt_1,173,617). coords(bww1gt_1,101,253,617).  visible(bww1gt_1,617). happensAt(walking(bww1gt_1),617). orientation(bww1gt_2,2,617). coords(bww1gt_2,142,206,617).  visible(bww1gt_2,617). happensAt(active(bww1gt_2),617).
orientation(bww1gt_1,173,618). coords(bww1gt_1,101,253,618).  visible(bww1gt_1,618). happensAt(walking(bww1gt_1),618). orientation(bww1gt_2,1,618). coords(bww1gt_2,142,206,618).  visible(bww1gt_2,618). happensAt(active(bww1gt_2),618).
orientation(bww1gt_1,173,619). coords(bww1gt_1,101,253,619).  visible(bww1gt_1,619). happensAt(walking(bww1gt_1),619). orientation(bww1gt_2,3,619). coords(bww1gt_2,143,207,619).  visible(bww1gt_2,619). happensAt(active(bww1gt_2),619).
orientation(bww1gt_1,173,620). coords(bww1gt_1,101,253,620).  visible(bww1gt_1,620). happensAt(walking(bww1gt_1),620). orientation(bww1gt_2,3,620). coords(bww1gt_2,144,206,620).  visible(bww1gt_2,620). happensAt(walking(bww1gt_2),620).
orientation(bww1gt_1,173,621). coords(bww1gt_1,101,254,621).  visible(bww1gt_1,621). happensAt(walking(bww1gt_1),621). orientation(bww1gt_2,178,621). coords(bww1gt_2,144,207,621).  visible(bww1gt_2,621). happensAt(walking(bww1gt_2),621).
orientation(bww1gt_1,173,622). coords(bww1gt_1,101,254,622).  visible(bww1gt_1,622). happensAt(walking(bww1gt_1),622). orientation(bww1gt_2,178,622). coords(bww1gt_2,145,207,622).  visible(bww1gt_2,622). happensAt(walking(bww1gt_2),622).
orientation(bww1gt_1,173,623). coords(bww1gt_1,101,254,623).  visible(bww1gt_1,623). happensAt(walking(bww1gt_1),623). orientation(bww1gt_2,177,623). coords(bww1gt_2,146,207,623).  visible(bww1gt_2,623). happensAt(walking(bww1gt_2),623).
orientation(bww1gt_1,173,624). coords(bww1gt_1,101,254,624).  visible(bww1gt_1,624). happensAt(walking(bww1gt_1),624). orientation(bww1gt_2,178,624). coords(bww1gt_2,146,207,624).  visible(bww1gt_2,624). happensAt(walking(bww1gt_2),624).
orientation(bww1gt_1,173,625). coords(bww1gt_1,101,254,625).  visible(bww1gt_1,625). happensAt(walking(bww1gt_1),625). orientation(bww1gt_2,169,625). coords(bww1gt_2,149,207,625).  visible(bww1gt_2,625). happensAt(walking(bww1gt_2),625).
orientation(bww1gt_1,173,626). coords(bww1gt_1,101,255,626).  visible(bww1gt_1,626). happensAt(walking(bww1gt_1),626). orientation(bww1gt_2,169,626). coords(bww1gt_2,152,208,626).  visible(bww1gt_2,626). happensAt(walking(bww1gt_2),626).
#show holdsAt/2.