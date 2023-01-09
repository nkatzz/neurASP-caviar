
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

	orientation(br4gt_0,112,127). coords(br4gt_0,302,248,127).  visible(br4gt_0,127). happensAt(walking(br4gt_0),127). orientation(br4gt_1,5,127). coords(br4gt_1,220,247,127).  visible(br4gt_1,127). happensAt(walking(br4gt_1),127).
orientation(br4gt_0,114,128). coords(br4gt_0,302,252,128).  visible(br4gt_0,128). happensAt(walking(br4gt_0),128). orientation(br4gt_1,3,128). coords(br4gt_1,219,245,128).  visible(br4gt_1,128). happensAt(walking(br4gt_1),128).
orientation(br4gt_0,123,129). coords(br4gt_0,303,256,129).  visible(br4gt_0,129). happensAt(walking(br4gt_0),129). orientation(br4gt_1,3,129). coords(br4gt_1,219,243,129).  visible(br4gt_1,129). happensAt(walking(br4gt_1),129).
orientation(br4gt_0,112,130). coords(br4gt_0,306,257,130).  visible(br4gt_0,130). happensAt(walking(br4gt_0),130). orientation(br4gt_1,2,130). coords(br4gt_1,218,241,130).  visible(br4gt_1,130). happensAt(walking(br4gt_1),130).
orientation(br4gt_0,40,131). coords(br4gt_0,307,259,131).  visible(br4gt_0,131). happensAt(walking(br4gt_0),131). orientation(br4gt_1,1,131). coords(br4gt_1,218,239,131).  visible(br4gt_1,131). happensAt(walking(br4gt_1),131).
orientation(br4gt_0,59,132). coords(br4gt_0,308,260,132).  visible(br4gt_0,132). happensAt(walking(br4gt_0),132). orientation(br4gt_1,179,132). coords(br4gt_1,217,236,132).  visible(br4gt_1,132). happensAt(walking(br4gt_1),132).
orientation(br4gt_0,59,133). coords(br4gt_0,309,261,133).  visible(br4gt_0,133). happensAt(walking(br4gt_0),133). orientation(br4gt_1,179,133). coords(br4gt_1,216,232,133).  visible(br4gt_1,133). happensAt(walking(br4gt_1),133).
orientation(br4gt_0,66,134). coords(br4gt_0,310,263,134).  visible(br4gt_0,134). happensAt(walking(br4gt_0),134). orientation(br4gt_1,179,134). coords(br4gt_1,215,229,134).  visible(br4gt_1,134). happensAt(walking(br4gt_1),134).
orientation(br4gt_0,56,135). coords(br4gt_0,311,264,135).  visible(br4gt_0,135). happensAt(walking(br4gt_0),135). orientation(br4gt_1,178,135). coords(br4gt_1,215,227,135).  visible(br4gt_1,135). happensAt(walking(br4gt_1),135).
orientation(br4gt_0,48,136). coords(br4gt_0,311,265,136).  visible(br4gt_0,136). happensAt(walking(br4gt_0),136). orientation(br4gt_1,159,136). coords(br4gt_1,215,227,136).  visible(br4gt_1,136). happensAt(walking(br4gt_1),136).
orientation(br4gt_0,49,137). coords(br4gt_0,312,267,137).  visible(br4gt_0,137). happensAt(walking(br4gt_0),137). orientation(br4gt_1,161,137). coords(br4gt_1,214,223,137).  visible(br4gt_1,137). happensAt(walking(br4gt_1),137).
orientation(br4gt_0,49,138). coords(br4gt_0,313,269,138).  visible(br4gt_0,138). happensAt(walking(br4gt_0),138). orientation(br4gt_1,161,138). coords(br4gt_1,213,222,138).  visible(br4gt_1,138). happensAt(walking(br4gt_1),138).
orientation(br4gt_0,59,139). coords(br4gt_0,313,271,139).  visible(br4gt_0,139). happensAt(walking(br4gt_0),139). orientation(br4gt_1,164,139). coords(br4gt_1,212,220,139).  visible(br4gt_1,139). happensAt(walking(br4gt_1),139).
orientation(br4gt_0,66,140). coords(br4gt_0,314,273,140).  visible(br4gt_0,140). happensAt(walking(br4gt_0),140). orientation(br4gt_1,151,140). coords(br4gt_1,212,219,140).  visible(br4gt_1,140). happensAt(walking(br4gt_1),140).
orientation(br4gt_0,61,141). coords(br4gt_0,314,275,141).  visible(br4gt_0,141). happensAt(walking(br4gt_0),141). orientation(br4gt_1,151,141). coords(br4gt_1,211,217,141).  visible(br4gt_1,141). happensAt(walking(br4gt_1),141).
orientation(br4gt_0,62,142). coords(br4gt_0,314,276,142).  visible(br4gt_0,142). happensAt(walking(br4gt_0),142). orientation(br4gt_1,151,142). coords(br4gt_1,210,215,142).  visible(br4gt_1,142). happensAt(walking(br4gt_1),142).
orientation(br4gt_0,57,143). coords(br4gt_0,315,276,143).  visible(br4gt_0,143). happensAt(walking(br4gt_0),143). orientation(br4gt_1,151,143). coords(br4gt_1,208,213,143).  visible(br4gt_1,143). happensAt(walking(br4gt_1),143).
orientation(br4gt_0,53,144). coords(br4gt_0,315,276,144).  visible(br4gt_0,144). happensAt(walking(br4gt_0),144). orientation(br4gt_1,151,144). coords(br4gt_1,206,210,144).  visible(br4gt_1,144). happensAt(walking(br4gt_1),144).
orientation(br4gt_0,56,145). coords(br4gt_0,315,276,145).  visible(br4gt_0,145). happensAt(walking(br4gt_0),145). orientation(br4gt_1,151,145). coords(br4gt_1,204,208,145).  visible(br4gt_1,145). happensAt(walking(br4gt_1),145).
orientation(br4gt_0,56,146). coords(br4gt_0,315,276,146).  visible(br4gt_0,146). happensAt(walking(br4gt_0),146). orientation(br4gt_1,151,146). coords(br4gt_1,202,206,146).  visible(br4gt_1,146). happensAt(walking(br4gt_1),146).
orientation(br4gt_0,43,147). coords(br4gt_0,315,276,147).  visible(br4gt_0,147). happensAt(walking(br4gt_0),147). orientation(br4gt_1,151,147). coords(br4gt_1,200,203,147).  visible(br4gt_1,147). happensAt(walking(br4gt_1),147).
orientation(br4gt_0,53,148). coords(br4gt_0,318,277,148).  visible(br4gt_0,148). happensAt(walking(br4gt_0),148). orientation(br4gt_1,151,148). coords(br4gt_1,198,202,148).  visible(br4gt_1,148). happensAt(walking(br4gt_1),148).
orientation(br4gt_0,82,149). coords(br4gt_0,323,279,149).  visible(br4gt_0,149). happensAt(walking(br4gt_0),149). orientation(br4gt_1,151,149). coords(br4gt_1,198,201,149).  visible(br4gt_1,149). happensAt(walking(br4gt_1),149).
orientation(br4gt_0,79,150). coords(br4gt_0,322,280,150).  visible(br4gt_0,150). happensAt(walking(br4gt_0),150). orientation(br4gt_1,151,150). coords(br4gt_1,197,198,150).  visible(br4gt_1,150). happensAt(walking(br4gt_1),150).
orientation(br4gt_0,90,151). coords(br4gt_0,324,282,151).  happensAt(disappear(br4gt_0),151). happensAt(walking(br4gt_0),151). orientation(br4gt_1,151,151). coords(br4gt_1,197,197,151).  visible(br4gt_1,151). happensAt(walking(br4gt_1),151).
#show holdsAt/2.