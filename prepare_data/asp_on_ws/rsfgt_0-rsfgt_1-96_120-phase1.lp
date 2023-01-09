
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

	orientation(rsfgt_0,160,96). coords(rsfgt_0,56,150,96).  visible(rsfgt_0,96). happensAt(walking(rsfgt_0),96). orientation(rsfgt_1,150,96). coords(rsfgt_1,165,103,96).  visible(rsfgt_1,96). happensAt(active(rsfgt_1),96).
orientation(rsfgt_0,160,97). coords(rsfgt_0,56,151,97).  visible(rsfgt_0,97). happensAt(walking(rsfgt_0),97). orientation(rsfgt_1,150,97). coords(rsfgt_1,165,103,97).  visible(rsfgt_1,97). happensAt(active(rsfgt_1),97).
orientation(rsfgt_0,160,98). coords(rsfgt_0,55,151,98).  visible(rsfgt_0,98). happensAt(walking(rsfgt_0),98). orientation(rsfgt_1,150,98). coords(rsfgt_1,165,103,98).  visible(rsfgt_1,98). happensAt(active(rsfgt_1),98).
orientation(rsfgt_0,160,99). coords(rsfgt_0,55,151,99).  visible(rsfgt_0,99). happensAt(walking(rsfgt_0),99). orientation(rsfgt_1,150,99). coords(rsfgt_1,165,103,99).  visible(rsfgt_1,99). happensAt(active(rsfgt_1),99).
orientation(rsfgt_0,160,100). coords(rsfgt_0,54,151,100).  visible(rsfgt_0,100). happensAt(walking(rsfgt_0),100). orientation(rsfgt_1,150,100). coords(rsfgt_1,165,103,100).  visible(rsfgt_1,100). happensAt(active(rsfgt_1),100).
orientation(rsfgt_0,160,101). coords(rsfgt_0,53,151,101).  visible(rsfgt_0,101). happensAt(walking(rsfgt_0),101). orientation(rsfgt_1,150,101). coords(rsfgt_1,165,103,101).  visible(rsfgt_1,101). happensAt(active(rsfgt_1),101).
orientation(rsfgt_0,160,102). coords(rsfgt_0,53,152,102).  visible(rsfgt_0,102). happensAt(walking(rsfgt_0),102). orientation(rsfgt_1,150,102). coords(rsfgt_1,165,103,102).  visible(rsfgt_1,102). happensAt(active(rsfgt_1),102).
orientation(rsfgt_0,160,103). coords(rsfgt_0,52,152,103).  visible(rsfgt_0,103). happensAt(walking(rsfgt_0),103). orientation(rsfgt_1,150,103). coords(rsfgt_1,165,104,103).  visible(rsfgt_1,103). happensAt(active(rsfgt_1),103).
orientation(rsfgt_0,160,104). coords(rsfgt_0,49,152,104).  visible(rsfgt_0,104). happensAt(walking(rsfgt_0),104). orientation(rsfgt_1,150,104). coords(rsfgt_1,165,104,104).  visible(rsfgt_1,104). happensAt(active(rsfgt_1),104).
orientation(rsfgt_0,160,105). coords(rsfgt_0,49,152,105).  visible(rsfgt_0,105). happensAt(walking(rsfgt_0),105). orientation(rsfgt_1,150,105). coords(rsfgt_1,165,104,105).  visible(rsfgt_1,105). happensAt(active(rsfgt_1),105).
orientation(rsfgt_0,160,106). coords(rsfgt_0,49,153,106).  visible(rsfgt_0,106). happensAt(walking(rsfgt_0),106). orientation(rsfgt_1,150,106). coords(rsfgt_1,166,106,106).  visible(rsfgt_1,106). happensAt(active(rsfgt_1),106).
orientation(rsfgt_0,160,107). coords(rsfgt_0,49,153,107).  visible(rsfgt_0,107). happensAt(walking(rsfgt_0),107). orientation(rsfgt_1,150,107). coords(rsfgt_1,166,106,107).  visible(rsfgt_1,107). happensAt(active(rsfgt_1),107).
orientation(rsfgt_0,160,108). coords(rsfgt_0,49,153,108).  visible(rsfgt_0,108). happensAt(walking(rsfgt_0),108). orientation(rsfgt_1,150,108). coords(rsfgt_1,166,106,108).  visible(rsfgt_1,108). happensAt(active(rsfgt_1),108).
orientation(rsfgt_0,160,109). coords(rsfgt_0,49,153,109).  visible(rsfgt_0,109). happensAt(walking(rsfgt_0),109). orientation(rsfgt_1,150,109). coords(rsfgt_1,166,106,109).  visible(rsfgt_1,109). happensAt(active(rsfgt_1),109).
orientation(rsfgt_0,160,110). coords(rsfgt_0,49,153,110).  visible(rsfgt_0,110). happensAt(walking(rsfgt_0),110). orientation(rsfgt_1,150,110). coords(rsfgt_1,166,106,110).  visible(rsfgt_1,110). happensAt(active(rsfgt_1),110).
orientation(rsfgt_0,160,111). coords(rsfgt_0,48,153,111).  visible(rsfgt_0,111). happensAt(walking(rsfgt_0),111). orientation(rsfgt_1,150,111). coords(rsfgt_1,166,106,111).  visible(rsfgt_1,111). happensAt(active(rsfgt_1),111).
orientation(rsfgt_0,160,112). coords(rsfgt_0,48,153,112).  visible(rsfgt_0,112). happensAt(walking(rsfgt_0),112). orientation(rsfgt_1,150,112). coords(rsfgt_1,166,106,112).  visible(rsfgt_1,112). happensAt(active(rsfgt_1),112).
orientation(rsfgt_0,160,113). coords(rsfgt_0,48,153,113).  visible(rsfgt_0,113). happensAt(walking(rsfgt_0),113). orientation(rsfgt_1,150,113). coords(rsfgt_1,166,106,113).  visible(rsfgt_1,113). happensAt(active(rsfgt_1),113).
orientation(rsfgt_0,160,114). coords(rsfgt_0,46,153,114).  visible(rsfgt_0,114). happensAt(walking(rsfgt_0),114). orientation(rsfgt_1,150,114). coords(rsfgt_1,166,106,114).  visible(rsfgt_1,114). happensAt(active(rsfgt_1),114).
orientation(rsfgt_0,160,115). coords(rsfgt_0,45,153,115).  visible(rsfgt_0,115). happensAt(walking(rsfgt_0),115). orientation(rsfgt_1,150,115). coords(rsfgt_1,166,106,115).  visible(rsfgt_1,115). happensAt(active(rsfgt_1),115).
orientation(rsfgt_0,160,116). coords(rsfgt_0,45,153,116).  visible(rsfgt_0,116). happensAt(walking(rsfgt_0),116). orientation(rsfgt_1,150,116). coords(rsfgt_1,166,106,116).  visible(rsfgt_1,116). happensAt(active(rsfgt_1),116).
orientation(rsfgt_0,160,117). coords(rsfgt_0,45,154,117).  visible(rsfgt_0,117). happensAt(walking(rsfgt_0),117). orientation(rsfgt_1,150,117). coords(rsfgt_1,166,106,117).  visible(rsfgt_1,117). happensAt(active(rsfgt_1),117).
orientation(rsfgt_0,160,118). coords(rsfgt_0,45,154,118).  visible(rsfgt_0,118). happensAt(walking(rsfgt_0),118). orientation(rsfgt_1,150,118). coords(rsfgt_1,166,106,118).  visible(rsfgt_1,118). happensAt(active(rsfgt_1),118).
orientation(rsfgt_0,160,119). coords(rsfgt_0,44,155,119).  visible(rsfgt_0,119). happensAt(walking(rsfgt_0),119). orientation(rsfgt_1,150,119). coords(rsfgt_1,166,106,119).  visible(rsfgt_1,119). happensAt(active(rsfgt_1),119).
orientation(rsfgt_0,160,120). coords(rsfgt_0,44,155,120).  visible(rsfgt_0,120). happensAt(walking(rsfgt_0),120). orientation(rsfgt_1,150,120). coords(rsfgt_1,166,106,120).  visible(rsfgt_1,120). happensAt(active(rsfgt_1),120).
#show holdsAt/2.