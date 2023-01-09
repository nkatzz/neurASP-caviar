
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

	orientation(rffgt_0,135,108). coords(rffgt_0,142,42,108).  visible(rffgt_0,108). happensAt(walking(rffgt_0),108). orientation(rffgt_1,146,108). coords(rffgt_1,152,109,108).  visible(rffgt_1,108). happensAt(inactive(rffgt_1),108).
orientation(rffgt_0,135,109). coords(rffgt_0,143,42,109).  visible(rffgt_0,109). happensAt(walking(rffgt_0),109). orientation(rffgt_1,146,109). coords(rffgt_1,152,110,109).  visible(rffgt_1,109). happensAt(inactive(rffgt_1),109).
orientation(rffgt_0,135,110). coords(rffgt_0,144,42,110).  visible(rffgt_0,110). happensAt(walking(rffgt_0),110). orientation(rffgt_1,146,110). coords(rffgt_1,152,109,110).  visible(rffgt_1,110). happensAt(inactive(rffgt_1),110).
orientation(rffgt_0,135,111). coords(rffgt_0,144,41,111).  visible(rffgt_0,111). happensAt(walking(rffgt_0),111). orientation(rffgt_1,146,111). coords(rffgt_1,152,110,111).  visible(rffgt_1,111). happensAt(inactive(rffgt_1),111).
orientation(rffgt_0,135,112). coords(rffgt_0,144,41,112).  visible(rffgt_0,112). happensAt(walking(rffgt_0),112). orientation(rffgt_1,146,112). coords(rffgt_1,152,110,112).  visible(rffgt_1,112). happensAt(inactive(rffgt_1),112).
orientation(rffgt_0,135,113). coords(rffgt_0,144,41,113).  visible(rffgt_0,113). happensAt(walking(rffgt_0),113). orientation(rffgt_1,146,113). coords(rffgt_1,152,110,113).  visible(rffgt_1,113). happensAt(inactive(rffgt_1),113).
orientation(rffgt_0,135,114). coords(rffgt_0,145,41,114).  visible(rffgt_0,114). happensAt(walking(rffgt_0),114). orientation(rffgt_1,146,114). coords(rffgt_1,152,110,114).  visible(rffgt_1,114). happensAt(inactive(rffgt_1),114).
orientation(rffgt_0,135,115). coords(rffgt_0,145,41,115).  visible(rffgt_0,115). happensAt(walking(rffgt_0),115). orientation(rffgt_1,146,115). coords(rffgt_1,151,109,115).  visible(rffgt_1,115). happensAt(inactive(rffgt_1),115).
orientation(rffgt_0,135,116). coords(rffgt_0,145,40,116).  visible(rffgt_0,116). happensAt(walking(rffgt_0),116). orientation(rffgt_1,146,116). coords(rffgt_1,151,109,116).  visible(rffgt_1,116). happensAt(inactive(rffgt_1),116).
orientation(rffgt_0,135,117). coords(rffgt_0,146,39,117).  visible(rffgt_0,117). happensAt(walking(rffgt_0),117). orientation(rffgt_1,146,117). coords(rffgt_1,151,109,117).  visible(rffgt_1,117). happensAt(inactive(rffgt_1),117).
orientation(rffgt_0,135,118). coords(rffgt_0,146,39,118).  visible(rffgt_0,118). happensAt(walking(rffgt_0),118). orientation(rffgt_1,146,118). coords(rffgt_1,152,110,118).  visible(rffgt_1,118). happensAt(inactive(rffgt_1),118).
orientation(rffgt_0,135,119). coords(rffgt_0,146,39,119).  visible(rffgt_0,119). happensAt(walking(rffgt_0),119). orientation(rffgt_1,146,119). coords(rffgt_1,152,110,119).  visible(rffgt_1,119). happensAt(inactive(rffgt_1),119).
orientation(rffgt_0,135,120). coords(rffgt_0,148,39,120).  visible(rffgt_0,120). happensAt(walking(rffgt_0),120). orientation(rffgt_1,146,120). coords(rffgt_1,152,110,120).  visible(rffgt_1,120). happensAt(inactive(rffgt_1),120).
orientation(rffgt_0,135,121). coords(rffgt_0,148,39,121).  visible(rffgt_0,121). happensAt(walking(rffgt_0),121). orientation(rffgt_1,146,121). coords(rffgt_1,152,109,121).  visible(rffgt_1,121). happensAt(inactive(rffgt_1),121).
orientation(rffgt_0,135,122). coords(rffgt_0,149,39,122).  visible(rffgt_0,122). happensAt(walking(rffgt_0),122). orientation(rffgt_1,146,122). coords(rffgt_1,152,109,122).  visible(rffgt_1,122). happensAt(inactive(rffgt_1),122).
orientation(rffgt_0,135,123). coords(rffgt_0,150,39,123).  visible(rffgt_0,123). happensAt(walking(rffgt_0),123). orientation(rffgt_1,148,123). coords(rffgt_1,152,109,123).  visible(rffgt_1,123). happensAt(inactive(rffgt_1),123).
orientation(rffgt_0,135,124). coords(rffgt_0,151,39,124).  visible(rffgt_0,124). happensAt(walking(rffgt_0),124). orientation(rffgt_1,148,124). coords(rffgt_1,152,109,124).  visible(rffgt_1,124). happensAt(inactive(rffgt_1),124).
orientation(rffgt_0,135,125). coords(rffgt_0,151,38,125).  visible(rffgt_0,125). happensAt(walking(rffgt_0),125). orientation(rffgt_1,148,125). coords(rffgt_1,152,109,125).  visible(rffgt_1,125). happensAt(inactive(rffgt_1),125).
orientation(rffgt_0,135,126). coords(rffgt_0,150,38,126).  visible(rffgt_0,126). happensAt(walking(rffgt_0),126). orientation(rffgt_1,148,126). coords(rffgt_1,152,109,126).  visible(rffgt_1,126). happensAt(inactive(rffgt_1),126).
orientation(rffgt_0,135,127). coords(rffgt_0,151,38,127).  visible(rffgt_0,127). happensAt(walking(rffgt_0),127). orientation(rffgt_1,148,127). coords(rffgt_1,152,109,127).  visible(rffgt_1,127). happensAt(inactive(rffgt_1),127).
orientation(rffgt_0,135,128). coords(rffgt_0,152,37,128).  visible(rffgt_0,128). happensAt(walking(rffgt_0),128). orientation(rffgt_1,148,128). coords(rffgt_1,152,109,128).  visible(rffgt_1,128). happensAt(inactive(rffgt_1),128).
orientation(rffgt_0,135,129). coords(rffgt_0,152,37,129).  visible(rffgt_0,129). happensAt(walking(rffgt_0),129). orientation(rffgt_1,148,129). coords(rffgt_1,152,109,129).  visible(rffgt_1,129). happensAt(inactive(rffgt_1),129).
orientation(rffgt_0,135,130). coords(rffgt_0,152,37,130).  visible(rffgt_0,130). happensAt(walking(rffgt_0),130). orientation(rffgt_1,148,130). coords(rffgt_1,152,109,130).  visible(rffgt_1,130). happensAt(inactive(rffgt_1),130).
orientation(rffgt_0,135,131). coords(rffgt_0,152,37,131).  visible(rffgt_0,131). happensAt(walking(rffgt_0),131). orientation(rffgt_1,148,131). coords(rffgt_1,152,109,131).  visible(rffgt_1,131). happensAt(inactive(rffgt_1),131).
orientation(rffgt_0,135,132). coords(rffgt_0,153,34,132).  visible(rffgt_0,132). happensAt(walking(rffgt_0),132). orientation(rffgt_1,148,132). coords(rffgt_1,152,109,132).  visible(rffgt_1,132). happensAt(inactive(rffgt_1),132).
#show holdsAt/2.