
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

	orientation(br1gt_1,167,36). coords(br1gt_1,144,202,36).  visible(br1gt_1,36). happensAt(walking(br1gt_1),36). orientation(br1gt_2,160,36). coords(br1gt_2,84,200,36).  visible(br1gt_2,36). happensAt(walking(br1gt_2),36).
orientation(br1gt_1,167,37). coords(br1gt_1,143,201,37).  visible(br1gt_1,37). happensAt(walking(br1gt_1),37). orientation(br1gt_2,169,37). coords(br1gt_2,85,201,37).  visible(br1gt_2,37). happensAt(walking(br1gt_2),37).
orientation(br1gt_1,167,38). coords(br1gt_1,142,201,38).  visible(br1gt_1,38). happensAt(walking(br1gt_1),38). orientation(br1gt_2,169,38). coords(br1gt_2,84,201,38).  visible(br1gt_2,38). happensAt(walking(br1gt_2),38).
orientation(br1gt_1,167,39). coords(br1gt_1,142,201,39).  visible(br1gt_1,39). happensAt(walking(br1gt_1),39). orientation(br1gt_2,169,39). coords(br1gt_2,85,202,39).  visible(br1gt_2,39). happensAt(walking(br1gt_2),39).
orientation(br1gt_1,167,40). coords(br1gt_1,141,200,40).  visible(br1gt_1,40). happensAt(walking(br1gt_1),40). orientation(br1gt_2,169,40). coords(br1gt_2,84,201,40).  visible(br1gt_2,40). happensAt(walking(br1gt_2),40).
orientation(br1gt_1,167,41). coords(br1gt_1,140,199,41).  visible(br1gt_1,41). happensAt(walking(br1gt_1),41). orientation(br1gt_2,169,41). coords(br1gt_2,85,204,41).  visible(br1gt_2,41). happensAt(walking(br1gt_2),41).
orientation(br1gt_1,167,42). coords(br1gt_1,139,197,42).  visible(br1gt_1,42). happensAt(walking(br1gt_1),42). orientation(br1gt_2,169,42). coords(br1gt_2,85,204,42).  visible(br1gt_2,42). happensAt(walking(br1gt_2),42).
orientation(br1gt_1,167,43). coords(br1gt_1,139,197,43).  visible(br1gt_1,43). happensAt(walking(br1gt_1),43). orientation(br1gt_2,169,43). coords(br1gt_2,86,204,43).  visible(br1gt_2,43). happensAt(walking(br1gt_2),43).
orientation(br1gt_1,167,44). coords(br1gt_1,135,197,44).  visible(br1gt_1,44). happensAt(walking(br1gt_1),44). orientation(br1gt_2,171,44). coords(br1gt_2,86,204,44).  visible(br1gt_2,44). happensAt(walking(br1gt_2),44).
orientation(br1gt_1,164,45). coords(br1gt_1,133,196,45).  visible(br1gt_1,45). happensAt(walking(br1gt_1),45). orientation(br1gt_2,171,45). coords(br1gt_2,85,204,45).  visible(br1gt_2,45). happensAt(walking(br1gt_2),45).
orientation(br1gt_1,164,46). coords(br1gt_1,130,195,46).  visible(br1gt_1,46). happensAt(walking(br1gt_1),46). orientation(br1gt_2,171,46). coords(br1gt_2,86,205,46).  visible(br1gt_2,46). happensAt(walking(br1gt_2),46).
orientation(br1gt_1,164,47). coords(br1gt_1,128,194,47).  visible(br1gt_1,47). happensAt(walking(br1gt_1),47). orientation(br1gt_2,171,47). coords(br1gt_2,86,205,47).  visible(br1gt_2,47). happensAt(walking(br1gt_2),47).
orientation(br1gt_1,164,48). coords(br1gt_1,125,193,48).  visible(br1gt_1,48). happensAt(walking(br1gt_1),48). orientation(br1gt_2,171,48). coords(br1gt_2,86,205,48).  visible(br1gt_2,48). happensAt(inactive(br1gt_2),48).
orientation(br1gt_1,164,49). coords(br1gt_1,124,192,49).  visible(br1gt_1,49). happensAt(walking(br1gt_1),49). orientation(br1gt_2,171,49). coords(br1gt_2,86,205,49).  visible(br1gt_2,49). happensAt(inactive(br1gt_2),49).
orientation(br1gt_1,164,50). coords(br1gt_1,124,191,50).  visible(br1gt_1,50). happensAt(walking(br1gt_1),50). orientation(br1gt_2,171,50). coords(br1gt_2,86,205,50).  visible(br1gt_2,50). happensAt(inactive(br1gt_2),50).
orientation(br1gt_1,164,51). coords(br1gt_1,123,190,51).  visible(br1gt_1,51). happensAt(walking(br1gt_1),51). orientation(br1gt_2,171,51). coords(br1gt_2,86,206,51).  visible(br1gt_2,51). happensAt(inactive(br1gt_2),51).
orientation(br1gt_1,164,52). coords(br1gt_1,123,189,52).  visible(br1gt_1,52). happensAt(walking(br1gt_1),52). orientation(br1gt_2,171,52). coords(br1gt_2,86,207,52).  visible(br1gt_2,52). happensAt(inactive(br1gt_2),52).
orientation(br1gt_1,164,53). coords(br1gt_1,122,188,53).  visible(br1gt_1,53). happensAt(walking(br1gt_1),53). orientation(br1gt_2,171,53). coords(br1gt_2,86,206,53).  visible(br1gt_2,53). happensAt(inactive(br1gt_2),53).
orientation(br1gt_1,164,54). coords(br1gt_1,121,188,54).  visible(br1gt_1,54). happensAt(walking(br1gt_1),54). orientation(br1gt_2,171,54). coords(br1gt_2,86,206,54).  visible(br1gt_2,54). happensAt(inactive(br1gt_2),54).
orientation(br1gt_1,164,55). coords(br1gt_1,120,187,55).  visible(br1gt_1,55). happensAt(walking(br1gt_1),55). orientation(br1gt_2,171,55). coords(br1gt_2,86,206,55).  visible(br1gt_2,55). happensAt(inactive(br1gt_2),55).
orientation(br1gt_1,164,56). coords(br1gt_1,118,187,56).  visible(br1gt_1,56). happensAt(walking(br1gt_1),56). orientation(br1gt_2,171,56). coords(br1gt_2,86,207,56).  visible(br1gt_2,56). happensAt(inactive(br1gt_2),56).
orientation(br1gt_1,164,57). coords(br1gt_1,117,186,57).  visible(br1gt_1,57). happensAt(walking(br1gt_1),57). orientation(br1gt_2,171,57). coords(br1gt_2,86,207,57).  visible(br1gt_2,57). happensAt(inactive(br1gt_2),57).
orientation(br1gt_1,164,58). coords(br1gt_1,116,186,58).  visible(br1gt_1,58). happensAt(walking(br1gt_1),58). orientation(br1gt_2,171,58). coords(br1gt_2,86,207,58).  visible(br1gt_2,58). happensAt(inactive(br1gt_2),58).
orientation(br1gt_1,164,59). coords(br1gt_1,111,186,59).  visible(br1gt_1,59). happensAt(walking(br1gt_1),59). orientation(br1gt_2,171,59). coords(br1gt_2,86,207,59).  visible(br1gt_2,59). happensAt(inactive(br1gt_2),59).
orientation(br1gt_1,164,60). coords(br1gt_1,109,185,60).  visible(br1gt_1,60). happensAt(walking(br1gt_1),60). orientation(br1gt_2,171,60). coords(br1gt_2,86,207,60).  visible(br1gt_2,60). happensAt(inactive(br1gt_2),60).
#show holdsAt/2.