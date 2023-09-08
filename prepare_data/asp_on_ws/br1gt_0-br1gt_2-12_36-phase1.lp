
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

	orientation(br1gt_0,175,12). coords(br1gt_0,267,262,12).  visible(br1gt_0,12). happensAt(walking(br1gt_0),12). orientation(br1gt_2,160,12). coords(br1gt_2,84,189,12).  visible(br1gt_2,12). happensAt(walking(br1gt_2),12).
orientation(br1gt_0,175,13). coords(br1gt_0,264,260,13).  visible(br1gt_0,13). happensAt(walking(br1gt_0),13). orientation(br1gt_2,160,13). coords(br1gt_2,84,191,13).  visible(br1gt_2,13). happensAt(walking(br1gt_2),13).
orientation(br1gt_0,6,14). coords(br1gt_0,260,257,14).  visible(br1gt_0,14). happensAt(walking(br1gt_0),14). orientation(br1gt_2,160,14). coords(br1gt_2,84,191,14).  visible(br1gt_2,14). happensAt(walking(br1gt_2),14).
orientation(br1gt_0,173,15). coords(br1gt_0,259,252,15).  visible(br1gt_0,15). happensAt(walking(br1gt_0),15). orientation(br1gt_2,160,15). coords(br1gt_2,84,193,15).  visible(br1gt_2,15). happensAt(walking(br1gt_2),15).
orientation(br1gt_0,169,16). coords(br1gt_0,258,249,16).  visible(br1gt_0,16). happensAt(walking(br1gt_0),16). orientation(br1gt_2,160,16). coords(br1gt_2,84,193,16).  visible(br1gt_2,16). happensAt(walking(br1gt_2),16).
orientation(br1gt_0,165,17). coords(br1gt_0,258,245,17).  visible(br1gt_0,17). happensAt(walking(br1gt_0),17). orientation(br1gt_2,160,17). coords(br1gt_2,84,194,17).  visible(br1gt_2,17). happensAt(walking(br1gt_2),17).
orientation(br1gt_0,164,18). coords(br1gt_0,256,242,18).  visible(br1gt_0,18). happensAt(walking(br1gt_0),18). orientation(br1gt_2,160,18). coords(br1gt_2,84,194,18).  visible(br1gt_2,18). happensAt(walking(br1gt_2),18).
orientation(br1gt_0,162,19). coords(br1gt_0,256,241,19).  visible(br1gt_0,19). happensAt(walking(br1gt_0),19). orientation(br1gt_2,160,19). coords(br1gt_2,84,195,19).  visible(br1gt_2,19). happensAt(walking(br1gt_2),19).
orientation(br1gt_0,159,20). coords(br1gt_0,255,241,20).  visible(br1gt_0,20). happensAt(walking(br1gt_0),20). orientation(br1gt_2,160,20). coords(br1gt_2,84,195,20).  visible(br1gt_2,20). happensAt(walking(br1gt_2),20).
orientation(br1gt_0,159,21). coords(br1gt_0,254,240,21).  visible(br1gt_0,21). happensAt(walking(br1gt_0),21). orientation(br1gt_2,160,21). coords(br1gt_2,84,195,21).  visible(br1gt_2,21). happensAt(walking(br1gt_2),21).
orientation(br1gt_0,164,22). coords(br1gt_0,249,239,22).  visible(br1gt_0,22). happensAt(walking(br1gt_0),22). orientation(br1gt_2,160,22). coords(br1gt_2,84,195,22).  visible(br1gt_2,22). happensAt(walking(br1gt_2),22).
orientation(br1gt_0,169,23). coords(br1gt_0,246,238,23).  visible(br1gt_0,23). happensAt(walking(br1gt_0),23). orientation(br1gt_2,160,23). coords(br1gt_2,84,196,23).  visible(br1gt_2,23). happensAt(walking(br1gt_2),23).
orientation(br1gt_0,174,24). coords(br1gt_0,243,234,24).  visible(br1gt_0,24). happensAt(walking(br1gt_0),24). orientation(br1gt_2,160,24). coords(br1gt_2,84,196,24).  visible(br1gt_2,24). happensAt(walking(br1gt_2),24).
orientation(br1gt_0,172,25). coords(br1gt_0,240,230,25).  visible(br1gt_0,25). happensAt(walking(br1gt_0),25). orientation(br1gt_2,160,25). coords(br1gt_2,83,196,25).  visible(br1gt_2,25). happensAt(walking(br1gt_2),25).
orientation(br1gt_0,172,26). coords(br1gt_0,237,225,26).  visible(br1gt_0,26). happensAt(walking(br1gt_0),26). orientation(br1gt_2,160,26). coords(br1gt_2,83,196,26).  visible(br1gt_2,26). happensAt(walking(br1gt_2),26).
orientation(br1gt_0,172,27). coords(br1gt_0,236,221,27).  visible(br1gt_0,27). happensAt(walking(br1gt_0),27). orientation(br1gt_2,160,27). coords(br1gt_2,83,196,27).  visible(br1gt_2,27). happensAt(walking(br1gt_2),27).
orientation(br1gt_0,172,28). coords(br1gt_0,235,216,28).  visible(br1gt_0,28). happensAt(walking(br1gt_0),28). orientation(br1gt_2,160,28). coords(br1gt_2,83,197,28).  visible(br1gt_2,28). happensAt(walking(br1gt_2),28).
orientation(br1gt_0,172,29). coords(br1gt_0,233,214,29).  visible(br1gt_0,29). happensAt(walking(br1gt_0),29). orientation(br1gt_2,160,29). coords(br1gt_2,84,199,29).  visible(br1gt_2,29). happensAt(walking(br1gt_2),29).
orientation(br1gt_0,172,30). coords(br1gt_0,232,211,30).  visible(br1gt_0,30). happensAt(walking(br1gt_0),30). orientation(br1gt_2,160,30). coords(br1gt_2,84,200,30).  visible(br1gt_2,30). happensAt(walking(br1gt_2),30).
orientation(br1gt_0,170,31). coords(br1gt_0,232,210,31).  visible(br1gt_0,31). happensAt(walking(br1gt_0),31). orientation(br1gt_2,160,31). coords(br1gt_2,83,197,31).  visible(br1gt_2,31). happensAt(walking(br1gt_2),31).
orientation(br1gt_0,170,32). coords(br1gt_0,233,209,32).  visible(br1gt_0,32). happensAt(walking(br1gt_0),32). orientation(br1gt_2,160,32). coords(br1gt_2,83,197,32).  visible(br1gt_2,32). happensAt(walking(br1gt_2),32).
orientation(br1gt_0,170,33). coords(br1gt_0,232,207,33).  visible(br1gt_0,33). happensAt(walking(br1gt_0),33). orientation(br1gt_2,160,33). coords(br1gt_2,84,199,33).  visible(br1gt_2,33). happensAt(walking(br1gt_2),33).
orientation(br1gt_0,166,34). coords(br1gt_0,232,206,34).  visible(br1gt_0,34). happensAt(walking(br1gt_0),34). orientation(br1gt_2,160,34). coords(br1gt_2,84,199,34).  visible(br1gt_2,34). happensAt(walking(br1gt_2),34).
orientation(br1gt_0,164,35). coords(br1gt_0,231,205,35).  visible(br1gt_0,35). happensAt(walking(br1gt_0),35). orientation(br1gt_2,160,35). coords(br1gt_2,84,200,35).  visible(br1gt_2,35). happensAt(walking(br1gt_2),35).
orientation(br1gt_0,162,36). coords(br1gt_0,229,203,36).  visible(br1gt_0,36). happensAt(walking(br1gt_0),36). orientation(br1gt_2,160,36). coords(br1gt_2,84,200,36).  visible(br1gt_2,36). happensAt(walking(br1gt_2),36).
#show holdsAt/2.