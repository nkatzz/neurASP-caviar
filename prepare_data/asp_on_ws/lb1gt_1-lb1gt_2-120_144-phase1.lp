
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

	orientation(lb1gt_1,151,120). coords(lb1gt_1,84,124,120).  visible(lb1gt_1,120). happensAt(walking(lb1gt_1),120). orientation(lb1gt_2,147,120). coords(lb1gt_2,94,96,120).  visible(lb1gt_2,120). happensAt(walking(lb1gt_2),120).
orientation(lb1gt_1,151,121). coords(lb1gt_1,84,124,121).  visible(lb1gt_1,121). happensAt(walking(lb1gt_1),121). orientation(lb1gt_2,147,121). coords(lb1gt_2,94,96,121).  visible(lb1gt_2,121). happensAt(walking(lb1gt_2),121).
orientation(lb1gt_1,151,122). coords(lb1gt_1,84,125,122).  visible(lb1gt_1,122). happensAt(walking(lb1gt_1),122). orientation(lb1gt_2,147,122). coords(lb1gt_2,94,97,122).  visible(lb1gt_2,122). happensAt(walking(lb1gt_2),122).
orientation(lb1gt_1,151,123). coords(lb1gt_1,84,126,123).  visible(lb1gt_1,123). happensAt(walking(lb1gt_1),123). orientation(lb1gt_2,147,123). coords(lb1gt_2,94,97,123).  visible(lb1gt_2,123). happensAt(walking(lb1gt_2),123).
orientation(lb1gt_1,151,124). coords(lb1gt_1,84,127,124).  visible(lb1gt_1,124). happensAt(walking(lb1gt_1),124). orientation(lb1gt_2,147,124). coords(lb1gt_2,94,98,124).  visible(lb1gt_2,124). happensAt(walking(lb1gt_2),124).
orientation(lb1gt_1,151,125). coords(lb1gt_1,84,129,125).  visible(lb1gt_1,125). happensAt(walking(lb1gt_1),125). orientation(lb1gt_2,147,125). coords(lb1gt_2,94,98,125).  visible(lb1gt_2,125). happensAt(walking(lb1gt_2),125).
orientation(lb1gt_1,151,126). coords(lb1gt_1,84,130,126).  visible(lb1gt_1,126). happensAt(walking(lb1gt_1),126). orientation(lb1gt_2,147,126). coords(lb1gt_2,94,98,126).  visible(lb1gt_2,126). happensAt(walking(lb1gt_2),126).
orientation(lb1gt_1,151,127). coords(lb1gt_1,85,131,127).  visible(lb1gt_1,127). happensAt(walking(lb1gt_1),127). orientation(lb1gt_2,144,127). coords(lb1gt_2,95,98,127).  visible(lb1gt_2,127). happensAt(walking(lb1gt_2),127).
orientation(lb1gt_1,151,128). coords(lb1gt_1,86,132,128).  visible(lb1gt_1,128). happensAt(walking(lb1gt_1),128). orientation(lb1gt_2,144,128). coords(lb1gt_2,95,98,128).  visible(lb1gt_2,128). happensAt(walking(lb1gt_2),128).
orientation(lb1gt_1,151,129). coords(lb1gt_1,86,132,129).  visible(lb1gt_1,129). happensAt(walking(lb1gt_1),129). orientation(lb1gt_2,144,129). coords(lb1gt_2,95,98,129).  visible(lb1gt_2,129). happensAt(walking(lb1gt_2),129).
orientation(lb1gt_1,151,130). coords(lb1gt_1,86,133,130).  visible(lb1gt_1,130). happensAt(walking(lb1gt_1),130). orientation(lb1gt_2,144,130). coords(lb1gt_2,95,99,130).  visible(lb1gt_2,130). happensAt(walking(lb1gt_2),130).
orientation(lb1gt_1,151,131). coords(lb1gt_1,86,133,131).  visible(lb1gt_1,131). happensAt(walking(lb1gt_1),131). orientation(lb1gt_2,144,131). coords(lb1gt_2,96,99,131).  visible(lb1gt_2,131). happensAt(walking(lb1gt_2),131).
orientation(lb1gt_1,151,132). coords(lb1gt_1,87,133,132).  visible(lb1gt_1,132). happensAt(walking(lb1gt_1),132). orientation(lb1gt_2,144,132). coords(lb1gt_2,96,99,132).  visible(lb1gt_2,132). happensAt(walking(lb1gt_2),132).
orientation(lb1gt_1,151,133). coords(lb1gt_1,87,133,133).  visible(lb1gt_1,133). happensAt(walking(lb1gt_1),133). orientation(lb1gt_2,144,133). coords(lb1gt_2,97,99,133).  visible(lb1gt_2,133). happensAt(walking(lb1gt_2),133).
orientation(lb1gt_1,151,134). coords(lb1gt_1,87,133,134).  visible(lb1gt_1,134). happensAt(walking(lb1gt_1),134). orientation(lb1gt_2,144,134). coords(lb1gt_2,97,99,134).  visible(lb1gt_2,134). happensAt(walking(lb1gt_2),134).
orientation(lb1gt_1,151,135). coords(lb1gt_1,87,134,135).  visible(lb1gt_1,135). happensAt(walking(lb1gt_1),135). orientation(lb1gt_2,144,135). coords(lb1gt_2,98,100,135).  visible(lb1gt_2,135). happensAt(walking(lb1gt_2),135).
orientation(lb1gt_1,151,136). coords(lb1gt_1,87,134,136).  visible(lb1gt_1,136). happensAt(walking(lb1gt_1),136). orientation(lb1gt_2,144,136). coords(lb1gt_2,98,100,136).  visible(lb1gt_2,136). happensAt(walking(lb1gt_2),136).
orientation(lb1gt_1,151,137). coords(lb1gt_1,87,134,137).  visible(lb1gt_1,137). happensAt(walking(lb1gt_1),137). orientation(lb1gt_2,144,137). coords(lb1gt_2,99,101,137).  visible(lb1gt_2,137). happensAt(walking(lb1gt_2),137).
orientation(lb1gt_1,151,138). coords(lb1gt_1,87,134,138).  visible(lb1gt_1,138). happensAt(walking(lb1gt_1),138). orientation(lb1gt_2,144,138). coords(lb1gt_2,99,102,138).  visible(lb1gt_2,138). happensAt(walking(lb1gt_2),138).
orientation(lb1gt_1,151,139). coords(lb1gt_1,87,134,139).  visible(lb1gt_1,139). happensAt(walking(lb1gt_1),139). orientation(lb1gt_2,144,139). coords(lb1gt_2,100,102,139).  visible(lb1gt_2,139). happensAt(walking(lb1gt_2),139).
orientation(lb1gt_1,151,140). coords(lb1gt_1,87,135,140).  visible(lb1gt_1,140). happensAt(walking(lb1gt_1),140). orientation(lb1gt_2,144,140). coords(lb1gt_2,100,102,140).  visible(lb1gt_2,140). happensAt(walking(lb1gt_2),140).
orientation(lb1gt_1,151,141). coords(lb1gt_1,88,135,141).  visible(lb1gt_1,141). happensAt(walking(lb1gt_1),141). orientation(lb1gt_2,144,141). coords(lb1gt_2,100,102,141).  visible(lb1gt_2,141). happensAt(walking(lb1gt_2),141).
orientation(lb1gt_1,151,142). coords(lb1gt_1,90,135,142).  visible(lb1gt_1,142). happensAt(walking(lb1gt_1),142). orientation(lb1gt_2,144,142). coords(lb1gt_2,100,102,142).  visible(lb1gt_2,142). happensAt(walking(lb1gt_2),142).
orientation(lb1gt_1,151,143). coords(lb1gt_1,90,135,143).  visible(lb1gt_1,143). happensAt(walking(lb1gt_1),143). orientation(lb1gt_2,144,143). coords(lb1gt_2,100,102,143).  visible(lb1gt_2,143). happensAt(walking(lb1gt_2),143).
orientation(lb1gt_1,151,144). coords(lb1gt_1,92,135,144).  visible(lb1gt_1,144). happensAt(walking(lb1gt_1),144). orientation(lb1gt_2,144,144). coords(lb1gt_2,100,102,144).  visible(lb1gt_2,144). happensAt(walking(lb1gt_2),144).
#show holdsAt/2.