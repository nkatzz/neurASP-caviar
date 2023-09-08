
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

	orientation(lb1gt_0,151,120). coords(lb1gt_0,180,120,120).  visible(lb1gt_0,120). happensAt(active(lb1gt_0),120). orientation(lb1gt_2,147,120). coords(lb1gt_2,94,96,120).  visible(lb1gt_2,120). happensAt(walking(lb1gt_2),120).
orientation(lb1gt_0,151,121). coords(lb1gt_0,180,120,121).  visible(lb1gt_0,121). happensAt(active(lb1gt_0),121). orientation(lb1gt_2,147,121). coords(lb1gt_2,94,96,121).  visible(lb1gt_2,121). happensAt(walking(lb1gt_2),121).
orientation(lb1gt_0,151,122). coords(lb1gt_0,180,120,122).  visible(lb1gt_0,122). happensAt(active(lb1gt_0),122). orientation(lb1gt_2,147,122). coords(lb1gt_2,94,97,122).  visible(lb1gt_2,122). happensAt(walking(lb1gt_2),122).
orientation(lb1gt_0,151,123). coords(lb1gt_0,180,119,123).  visible(lb1gt_0,123). happensAt(active(lb1gt_0),123). orientation(lb1gt_2,147,123). coords(lb1gt_2,94,97,123).  visible(lb1gt_2,123). happensAt(walking(lb1gt_2),123).
orientation(lb1gt_0,149,124). coords(lb1gt_0,180,119,124).  visible(lb1gt_0,124). happensAt(active(lb1gt_0),124). orientation(lb1gt_2,147,124). coords(lb1gt_2,94,98,124).  visible(lb1gt_2,124). happensAt(walking(lb1gt_2),124).
orientation(lb1gt_0,146,125). coords(lb1gt_0,180,119,125).  visible(lb1gt_0,125). happensAt(active(lb1gt_0),125). orientation(lb1gt_2,147,125). coords(lb1gt_2,94,98,125).  visible(lb1gt_2,125). happensAt(walking(lb1gt_2),125).
orientation(lb1gt_0,145,126). coords(lb1gt_0,180,117,126).  visible(lb1gt_0,126). happensAt(walking(lb1gt_0),126). orientation(lb1gt_2,147,126). coords(lb1gt_2,94,98,126).  visible(lb1gt_2,126). happensAt(walking(lb1gt_2),126).
orientation(lb1gt_0,145,127). coords(lb1gt_0,179,116,127).  visible(lb1gt_0,127). happensAt(walking(lb1gt_0),127). orientation(lb1gt_2,144,127). coords(lb1gt_2,95,98,127).  visible(lb1gt_2,127). happensAt(walking(lb1gt_2),127).
orientation(lb1gt_0,144,128). coords(lb1gt_0,179,115,128).  visible(lb1gt_0,128). happensAt(walking(lb1gt_0),128). orientation(lb1gt_2,144,128). coords(lb1gt_2,95,98,128).  visible(lb1gt_2,128). happensAt(walking(lb1gt_2),128).
orientation(lb1gt_0,143,129). coords(lb1gt_0,179,114,129).  visible(lb1gt_0,129). happensAt(walking(lb1gt_0),129). orientation(lb1gt_2,144,129). coords(lb1gt_2,95,98,129).  visible(lb1gt_2,129). happensAt(walking(lb1gt_2),129).
orientation(lb1gt_0,142,130). coords(lb1gt_0,179,113,130).  visible(lb1gt_0,130). happensAt(walking(lb1gt_0),130). orientation(lb1gt_2,144,130). coords(lb1gt_2,95,99,130).  visible(lb1gt_2,130). happensAt(walking(lb1gt_2),130).
orientation(lb1gt_0,142,131). coords(lb1gt_0,178,112,131).  visible(lb1gt_0,131). happensAt(walking(lb1gt_0),131). orientation(lb1gt_2,144,131). coords(lb1gt_2,96,99,131).  visible(lb1gt_2,131). happensAt(walking(lb1gt_2),131).
orientation(lb1gt_0,142,132). coords(lb1gt_0,177,111,132).  visible(lb1gt_0,132). happensAt(walking(lb1gt_0),132). orientation(lb1gt_2,144,132). coords(lb1gt_2,96,99,132).  visible(lb1gt_2,132). happensAt(walking(lb1gt_2),132).
orientation(lb1gt_0,142,133). coords(lb1gt_0,176,110,133).  visible(lb1gt_0,133). happensAt(walking(lb1gt_0),133). orientation(lb1gt_2,144,133). coords(lb1gt_2,97,99,133).  visible(lb1gt_2,133). happensAt(walking(lb1gt_2),133).
orientation(lb1gt_0,142,134). coords(lb1gt_0,175,109,134).  visible(lb1gt_0,134). happensAt(walking(lb1gt_0),134). orientation(lb1gt_2,144,134). coords(lb1gt_2,97,99,134).  visible(lb1gt_2,134). happensAt(walking(lb1gt_2),134).
orientation(lb1gt_0,142,135). coords(lb1gt_0,173,108,135).  visible(lb1gt_0,135). happensAt(walking(lb1gt_0),135). orientation(lb1gt_2,144,135). coords(lb1gt_2,98,100,135).  visible(lb1gt_2,135). happensAt(walking(lb1gt_2),135).
orientation(lb1gt_0,142,136). coords(lb1gt_0,173,107,136).  visible(lb1gt_0,136). happensAt(walking(lb1gt_0),136). orientation(lb1gt_2,144,136). coords(lb1gt_2,98,100,136).  visible(lb1gt_2,136). happensAt(walking(lb1gt_2),136).
orientation(lb1gt_0,142,137). coords(lb1gt_0,173,106,137).  visible(lb1gt_0,137). happensAt(walking(lb1gt_0),137). orientation(lb1gt_2,144,137). coords(lb1gt_2,99,101,137).  visible(lb1gt_2,137). happensAt(walking(lb1gt_2),137).
orientation(lb1gt_0,142,138). coords(lb1gt_0,173,104,138).  visible(lb1gt_0,138). happensAt(walking(lb1gt_0),138). orientation(lb1gt_2,144,138). coords(lb1gt_2,99,102,138).  visible(lb1gt_2,138). happensAt(walking(lb1gt_2),138).
orientation(lb1gt_0,138,139). coords(lb1gt_0,172,101,139).  visible(lb1gt_0,139). happensAt(walking(lb1gt_0),139). orientation(lb1gt_2,144,139). coords(lb1gt_2,100,102,139).  visible(lb1gt_2,139). happensAt(walking(lb1gt_2),139).
orientation(lb1gt_0,139,140). coords(lb1gt_0,172,98,140).  visible(lb1gt_0,140). happensAt(walking(lb1gt_0),140). orientation(lb1gt_2,144,140). coords(lb1gt_2,100,102,140).  visible(lb1gt_2,140). happensAt(walking(lb1gt_2),140).
orientation(lb1gt_0,138,141). coords(lb1gt_0,171,98,141).  visible(lb1gt_0,141). happensAt(walking(lb1gt_0),141). orientation(lb1gt_2,144,141). coords(lb1gt_2,100,102,141).  visible(lb1gt_2,141). happensAt(walking(lb1gt_2),141).
orientation(lb1gt_0,136,142). coords(lb1gt_0,171,96,142).  visible(lb1gt_0,142). happensAt(walking(lb1gt_0),142). orientation(lb1gt_2,144,142). coords(lb1gt_2,100,102,142).  visible(lb1gt_2,142). happensAt(walking(lb1gt_2),142).
orientation(lb1gt_0,136,143). coords(lb1gt_0,170,95,143).  visible(lb1gt_0,143). happensAt(walking(lb1gt_0),143). orientation(lb1gt_2,144,143). coords(lb1gt_2,100,102,143).  visible(lb1gt_2,143). happensAt(walking(lb1gt_2),143).
orientation(lb1gt_0,136,144). coords(lb1gt_0,170,93,144).  visible(lb1gt_0,144). happensAt(walking(lb1gt_0),144). orientation(lb1gt_2,144,144). coords(lb1gt_2,100,102,144).  visible(lb1gt_2,144). happensAt(walking(lb1gt_2),144).
#show holdsAt/2.