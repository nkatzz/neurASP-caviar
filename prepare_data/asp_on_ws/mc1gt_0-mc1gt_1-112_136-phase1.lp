
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

	orientation(mc1gt_0,147,112). coords(mc1gt_0,82,101,112).  visible(mc1gt_0,112). happensAt(walking(mc1gt_0),112). orientation(mc1gt_1,145,112). coords(mc1gt_1,100,82,112).  visible(mc1gt_1,112). happensAt(walking(mc1gt_1),112).
orientation(mc1gt_0,147,113). coords(mc1gt_0,82,101,113).  visible(mc1gt_0,113). happensAt(walking(mc1gt_0),113). orientation(mc1gt_1,145,113). coords(mc1gt_1,101,82,113).  visible(mc1gt_1,113). happensAt(walking(mc1gt_1),113).
orientation(mc1gt_0,147,114). coords(mc1gt_0,82,101,114).  visible(mc1gt_0,114). happensAt(walking(mc1gt_0),114). orientation(mc1gt_1,145,114). coords(mc1gt_1,102,82,114).  visible(mc1gt_1,114). happensAt(walking(mc1gt_1),114).
orientation(mc1gt_0,147,115). coords(mc1gt_0,82,101,115).  visible(mc1gt_0,115). happensAt(walking(mc1gt_0),115). orientation(mc1gt_1,145,115). coords(mc1gt_1,102,82,115).  visible(mc1gt_1,115). happensAt(walking(mc1gt_1),115).
orientation(mc1gt_0,147,116). coords(mc1gt_0,82,101,116).  visible(mc1gt_0,116). happensAt(walking(mc1gt_0),116). orientation(mc1gt_1,145,116). coords(mc1gt_1,102,82,116).  visible(mc1gt_1,116). happensAt(walking(mc1gt_1),116).
orientation(mc1gt_0,147,117). coords(mc1gt_0,82,101,117).  visible(mc1gt_0,117). happensAt(walking(mc1gt_0),117). orientation(mc1gt_1,145,117). coords(mc1gt_1,102,82,117).  visible(mc1gt_1,117). happensAt(walking(mc1gt_1),117).
orientation(mc1gt_0,147,118). coords(mc1gt_0,83,102,118).  visible(mc1gt_0,118). happensAt(walking(mc1gt_0),118). orientation(mc1gt_1,145,118). coords(mc1gt_1,102,82,118).  visible(mc1gt_1,118). happensAt(walking(mc1gt_1),118).
orientation(mc1gt_0,147,119). coords(mc1gt_0,83,102,119).  visible(mc1gt_0,119). happensAt(walking(mc1gt_0),119). orientation(mc1gt_1,145,119). coords(mc1gt_1,102,82,119).  visible(mc1gt_1,119). happensAt(walking(mc1gt_1),119).
orientation(mc1gt_0,147,120). coords(mc1gt_0,84,102,120).  visible(mc1gt_0,120). happensAt(walking(mc1gt_0),120). orientation(mc1gt_1,145,120). coords(mc1gt_1,103,82,120).  visible(mc1gt_1,120). happensAt(walking(mc1gt_1),120).
orientation(mc1gt_0,147,121). coords(mc1gt_0,85,102,121).  visible(mc1gt_0,121). happensAt(walking(mc1gt_0),121). orientation(mc1gt_1,145,121). coords(mc1gt_1,103,82,121).  visible(mc1gt_1,121). happensAt(walking(mc1gt_1),121).
orientation(mc1gt_0,147,122). coords(mc1gt_0,86,102,122).  visible(mc1gt_0,122). happensAt(walking(mc1gt_0),122). orientation(mc1gt_1,145,122). coords(mc1gt_1,105,82,122).  visible(mc1gt_1,122). happensAt(walking(mc1gt_1),122).
orientation(mc1gt_0,147,123). coords(mc1gt_0,87,102,123).  visible(mc1gt_0,123). happensAt(walking(mc1gt_0),123). orientation(mc1gt_1,145,123). coords(mc1gt_1,106,82,123).  visible(mc1gt_1,123). happensAt(walking(mc1gt_1),123).
orientation(mc1gt_0,147,124). coords(mc1gt_0,87,102,124).  visible(mc1gt_0,124). happensAt(walking(mc1gt_0),124). orientation(mc1gt_1,145,124). coords(mc1gt_1,107,82,124).  visible(mc1gt_1,124). happensAt(walking(mc1gt_1),124).
orientation(mc1gt_0,147,125). coords(mc1gt_0,87,102,125).  visible(mc1gt_0,125). happensAt(walking(mc1gt_0),125). orientation(mc1gt_1,145,125). coords(mc1gt_1,108,82,125).  visible(mc1gt_1,125). happensAt(walking(mc1gt_1),125).
orientation(mc1gt_0,147,126). coords(mc1gt_0,88,103,126).  visible(mc1gt_0,126). happensAt(walking(mc1gt_0),126). orientation(mc1gt_1,145,126). coords(mc1gt_1,109,82,126).  visible(mc1gt_1,126). happensAt(walking(mc1gt_1),126).
orientation(mc1gt_0,147,127). coords(mc1gt_0,89,103,127).  visible(mc1gt_0,127). happensAt(walking(mc1gt_0),127). orientation(mc1gt_1,145,127). coords(mc1gt_1,110,82,127).  visible(mc1gt_1,127). happensAt(walking(mc1gt_1),127).
orientation(mc1gt_0,147,128). coords(mc1gt_0,89,103,128).  visible(mc1gt_0,128). happensAt(walking(mc1gt_0),128). orientation(mc1gt_1,145,128). coords(mc1gt_1,110,82,128).  visible(mc1gt_1,128). happensAt(walking(mc1gt_1),128).
orientation(mc1gt_0,147,129). coords(mc1gt_0,89,103,129).  visible(mc1gt_0,129). happensAt(walking(mc1gt_0),129). orientation(mc1gt_1,145,129). coords(mc1gt_1,110,82,129).  visible(mc1gt_1,129). happensAt(walking(mc1gt_1),129).
orientation(mc1gt_0,147,130). coords(mc1gt_0,89,103,130).  visible(mc1gt_0,130). happensAt(walking(mc1gt_0),130). orientation(mc1gt_1,145,130). coords(mc1gt_1,111,82,130).  visible(mc1gt_1,130). happensAt(walking(mc1gt_1),130).
orientation(mc1gt_0,147,131). coords(mc1gt_0,89,104,131).  visible(mc1gt_0,131). happensAt(walking(mc1gt_0),131). orientation(mc1gt_1,145,131). coords(mc1gt_1,111,81,131).  visible(mc1gt_1,131). happensAt(walking(mc1gt_1),131).
orientation(mc1gt_0,147,132). coords(mc1gt_0,89,104,132).  visible(mc1gt_0,132). happensAt(walking(mc1gt_0),132). orientation(mc1gt_1,145,132). coords(mc1gt_1,112,81,132).  visible(mc1gt_1,132). happensAt(walking(mc1gt_1),132).
orientation(mc1gt_0,147,133). coords(mc1gt_0,90,105,133).  visible(mc1gt_0,133). happensAt(walking(mc1gt_0),133). orientation(mc1gt_1,145,133). coords(mc1gt_1,112,81,133).  visible(mc1gt_1,133). happensAt(walking(mc1gt_1),133).
orientation(mc1gt_0,147,134). coords(mc1gt_0,90,105,134).  visible(mc1gt_0,134). happensAt(walking(mc1gt_0),134). orientation(mc1gt_1,145,134). coords(mc1gt_1,112,81,134).  visible(mc1gt_1,134). happensAt(walking(mc1gt_1),134).
orientation(mc1gt_0,147,135). coords(mc1gt_0,91,106,135).  visible(mc1gt_0,135). happensAt(walking(mc1gt_0),135). orientation(mc1gt_1,145,135). coords(mc1gt_1,113,81,135).  visible(mc1gt_1,135). happensAt(walking(mc1gt_1),135).
orientation(mc1gt_0,147,136). coords(mc1gt_0,92,106,136).  visible(mc1gt_0,136). happensAt(walking(mc1gt_0),136). orientation(mc1gt_1,145,136). coords(mc1gt_1,113,81,136).  visible(mc1gt_1,136). happensAt(walking(mc1gt_1),136).
#show holdsAt/2.