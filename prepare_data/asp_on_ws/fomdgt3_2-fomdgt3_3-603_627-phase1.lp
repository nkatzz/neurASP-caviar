
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

	orientation(fomdgt3_2,130,603). coords(fomdgt3_2,253,117,603).  visible(fomdgt3_2,603). happensAt(walking(fomdgt3_2),603). orientation(fomdgt3_3,159,603). coords(fomdgt3_3,60,98,603).  visible(fomdgt3_3,603). happensAt(walking(fomdgt3_3),603).
orientation(fomdgt3_2,130,604). coords(fomdgt3_2,253,119,604).  visible(fomdgt3_2,604). happensAt(walking(fomdgt3_2),604). orientation(fomdgt3_3,159,604). coords(fomdgt3_3,60,98,604).  visible(fomdgt3_3,604). happensAt(walking(fomdgt3_3),604).
orientation(fomdgt3_2,130,605). coords(fomdgt3_2,253,122,605).  visible(fomdgt3_2,605). happensAt(walking(fomdgt3_2),605). orientation(fomdgt3_3,159,605). coords(fomdgt3_3,60,98,605).  visible(fomdgt3_3,605). happensAt(walking(fomdgt3_3),605).
orientation(fomdgt3_2,130,606). coords(fomdgt3_2,253,125,606).  visible(fomdgt3_2,606). happensAt(walking(fomdgt3_2),606). orientation(fomdgt3_3,159,606). coords(fomdgt3_3,63,98,606).  visible(fomdgt3_3,606). happensAt(walking(fomdgt3_3),606).
orientation(fomdgt3_2,130,607). coords(fomdgt3_2,253,127,607).  visible(fomdgt3_2,607). happensAt(walking(fomdgt3_2),607). orientation(fomdgt3_3,159,607). coords(fomdgt3_3,64,98,607).  visible(fomdgt3_3,607). happensAt(walking(fomdgt3_3),607).
orientation(fomdgt3_2,130,608). coords(fomdgt3_2,253,128,608).  visible(fomdgt3_2,608). happensAt(walking(fomdgt3_2),608). orientation(fomdgt3_3,159,608). coords(fomdgt3_3,65,97,608).  visible(fomdgt3_3,608). happensAt(walking(fomdgt3_3),608).
orientation(fomdgt3_2,130,609). coords(fomdgt3_2,253,130,609).  visible(fomdgt3_2,609). happensAt(walking(fomdgt3_2),609). orientation(fomdgt3_3,159,609). coords(fomdgt3_3,65,96,609).  visible(fomdgt3_3,609). happensAt(walking(fomdgt3_3),609).
orientation(fomdgt3_2,130,610). coords(fomdgt3_2,253,131,610).  visible(fomdgt3_2,610). happensAt(walking(fomdgt3_2),610). orientation(fomdgt3_3,159,610). coords(fomdgt3_3,65,96,610).  visible(fomdgt3_3,610). happensAt(walking(fomdgt3_3),610).
orientation(fomdgt3_2,130,611). coords(fomdgt3_2,253,132,611).  visible(fomdgt3_2,611). happensAt(walking(fomdgt3_2),611). orientation(fomdgt3_3,159,611). coords(fomdgt3_3,65,96,611).  visible(fomdgt3_3,611). happensAt(walking(fomdgt3_3),611).
orientation(fomdgt3_2,130,612). coords(fomdgt3_2,253,133,612).  visible(fomdgt3_2,612). happensAt(walking(fomdgt3_2),612). orientation(fomdgt3_3,159,612). coords(fomdgt3_3,65,96,612).  visible(fomdgt3_3,612). happensAt(walking(fomdgt3_3),612).
orientation(fomdgt3_2,130,613). coords(fomdgt3_2,253,135,613).  visible(fomdgt3_2,613). happensAt(walking(fomdgt3_2),613). orientation(fomdgt3_3,159,613). coords(fomdgt3_3,66,94,613).  visible(fomdgt3_3,613). happensAt(walking(fomdgt3_3),613).
orientation(fomdgt3_2,130,614). coords(fomdgt3_2,253,135,614).  visible(fomdgt3_2,614). happensAt(walking(fomdgt3_2),614). orientation(fomdgt3_3,159,614). coords(fomdgt3_3,66,94,614).  visible(fomdgt3_3,614). happensAt(walking(fomdgt3_3),614).
orientation(fomdgt3_2,130,615). coords(fomdgt3_2,253,135,615).  visible(fomdgt3_2,615). happensAt(walking(fomdgt3_2),615). orientation(fomdgt3_3,159,615). coords(fomdgt3_3,69,95,615).  visible(fomdgt3_3,615). happensAt(walking(fomdgt3_3),615).
orientation(fomdgt3_2,130,616). coords(fomdgt3_2,253,135,616).  visible(fomdgt3_2,616). happensAt(walking(fomdgt3_2),616). orientation(fomdgt3_3,159,616). coords(fomdgt3_3,69,95,616).  visible(fomdgt3_3,616). happensAt(walking(fomdgt3_3),616).
orientation(fomdgt3_2,130,617). coords(fomdgt3_2,256,136,617).  visible(fomdgt3_2,617). happensAt(walking(fomdgt3_2),617). orientation(fomdgt3_3,159,617). coords(fomdgt3_3,69,95,617).  visible(fomdgt3_3,617). happensAt(walking(fomdgt3_3),617).
orientation(fomdgt3_2,130,618). coords(fomdgt3_2,256,137,618).  visible(fomdgt3_2,618). happensAt(walking(fomdgt3_2),618). orientation(fomdgt3_3,159,618). coords(fomdgt3_3,71,95,618).  visible(fomdgt3_3,618). happensAt(walking(fomdgt3_3),618).
orientation(fomdgt3_2,130,619). coords(fomdgt3_2,258,139,619).  visible(fomdgt3_2,619). happensAt(walking(fomdgt3_2),619). orientation(fomdgt3_3,159,619). coords(fomdgt3_3,71,95,619).  visible(fomdgt3_3,619). happensAt(walking(fomdgt3_3),619).
orientation(fomdgt3_2,130,620). coords(fomdgt3_2,259,142,620).  visible(fomdgt3_2,620). happensAt(walking(fomdgt3_2),620). orientation(fomdgt3_3,159,620). coords(fomdgt3_3,71,95,620).  visible(fomdgt3_3,620). happensAt(walking(fomdgt3_3),620).
orientation(fomdgt3_2,130,621). coords(fomdgt3_2,260,145,621).  visible(fomdgt3_2,621). happensAt(walking(fomdgt3_2),621). orientation(fomdgt3_3,159,621). coords(fomdgt3_3,71,95,621).  visible(fomdgt3_3,621). happensAt(walking(fomdgt3_3),621).
orientation(fomdgt3_2,130,622). coords(fomdgt3_2,260,148,622).  visible(fomdgt3_2,622). happensAt(walking(fomdgt3_2),622). orientation(fomdgt3_3,159,622). coords(fomdgt3_3,71,95,622).  visible(fomdgt3_3,622). happensAt(walking(fomdgt3_3),622).
orientation(fomdgt3_2,130,623). coords(fomdgt3_2,261,151,623).  visible(fomdgt3_2,623). happensAt(walking(fomdgt3_2),623). orientation(fomdgt3_3,159,623). coords(fomdgt3_3,71,95,623).  visible(fomdgt3_3,623). happensAt(walking(fomdgt3_3),623).
orientation(fomdgt3_2,130,624). coords(fomdgt3_2,262,152,624).  visible(fomdgt3_2,624). happensAt(walking(fomdgt3_2),624). orientation(fomdgt3_3,159,624). coords(fomdgt3_3,71,95,624).  visible(fomdgt3_3,624). happensAt(walking(fomdgt3_3),624).
orientation(fomdgt3_2,130,625). coords(fomdgt3_2,263,154,625).  visible(fomdgt3_2,625). happensAt(walking(fomdgt3_2),625). orientation(fomdgt3_3,159,625). coords(fomdgt3_3,71,95,625).  visible(fomdgt3_3,625). happensAt(walking(fomdgt3_3),625).
orientation(fomdgt3_2,130,626). coords(fomdgt3_2,263,155,626).  visible(fomdgt3_2,626). happensAt(walking(fomdgt3_2),626). orientation(fomdgt3_3,159,626). coords(fomdgt3_3,71,95,626).  visible(fomdgt3_3,626). happensAt(walking(fomdgt3_3),626).
orientation(fomdgt3_2,130,627). coords(fomdgt3_2,263,156,627).  visible(fomdgt3_2,627). happensAt(walking(fomdgt3_2),627). orientation(fomdgt3_3,159,627). coords(fomdgt3_3,71,95,627).  visible(fomdgt3_3,627). happensAt(walking(fomdgt3_3),627).
#show holdsAt/2.